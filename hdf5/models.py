#    Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
#    Written by Nikolaos Pappas <nikolaos.pappas@idiap.ch>,
#
#    This file is part of gile.
#
#    gile is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3 as
#    published by the Free Software Foundation.
#
#    gile is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with gile. If not, see http://www.gnu.org/licenses/

import os, sys
import numpy as np
import time, theano, json, pickle
from util import load_vectors, load_meanyvecc
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.core import Lambda, Reshape, Layer
from keras.layers.merge import Multiply, Concatenate
from keras.layers import Input, TimeDistributed, Dense, GRU, merge, Add, Dropout
from keras.layers import Permute, RepeatVector, Flatten, Activation, Embedding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import label_ranking_loss as rankloss
from sklearn.metrics import label_ranking_average_precision_score as avgprec
from keras.callbacks import Callback
from util import load_word_vectors, one_error


class ValidateCallback(Callback):
	"""Class responsible for defining the function to be called during validation. """

	def __init__(self, X_val, Y_val, labels, mhan):
		""" Initialization. """
		super(ValidateCallback, self).__init__()
		self.X_val = X_val
		self.Y_val = Y_val
		self.labels = labels,
		self.mhan = mhan
		self.args = mhan.args
		self.bsl_vecs = np.array([self.labels[0] for j in range(self.mhan.args['bs'])])

	def on_epoch_end(self, epoch, logs={}):
		""" Callback. """
		val_score = self.mhan.eval(0, self.X_val ,self.Y_val, self.bsl_vecs, bs=self.args['bs'])
		self.mhan.save_model(0, epoch, 0.0, val_score)


class DataGenerator(object):
	"""Class responsible for loading data and generating batches for training. """

	def __init__(self, X, Y, l_vecs, wpad, bs, sampling, model, chunk_mode=True):
	  """ Initialization. """
	  self.X = X
	  self.Y = Y
	  self.l_vecs = np.array(l_vecs)
	  self.wpad = wpad
	  self.bs = bs
	  self.sampling = sampling
	  self.init_ids = np.arange(len(l_vecs))
	  self.model = model
	  self.chunk_mode = chunk_mode
	  if self.model.args['la']:
	  	self.bsl_vecs = np.array([self.l_vecs for j in range(self.model.args['bs'])])

	def generate(self, indexes):
		""" Generates batches. """
		while True:
			chunk_id = 0
			imax = int(len(indexes)/self.bs)
			for i in range(imax):
				cur_idxs = indexes[i*self.bs:(i+1)*self.bs]
				X, Y = self.__data_generation(chunk_id, cur_idxs)
				if self.model.args['la'] and len(X[0]) < self.bs:
					chunk_id += 1
					print " / chunk=%d " % chunk_id
				elif not self.model.args['la'] and len(X) < self.bs:
					chunk_id +=1
					print " / chunk=%d " % chunk_id
				if self.model.args['la']:
					yield [X[0], X[1]], Y
				else:
					yield X, Y


	def __data_generation(self, chunk_id, idxs):
		""" Load data for a particular chunk id with or without sampling. """
		if self.chunk_mode:
			x_vecs, y_vecs = load_vectors(self.X[chunk_id], self.Y[chunk_id], idxs, self.wpad, len(self.l_vecs), self.model)
		else:
			x_vecs, y_vecs = load_vectors(self.X, self.Y, idxs, self.wpad, len(self.l_vecs), self.model)
		if self.model.args['la']:
			if self.sampling < 1.0:
				ls_vecs, ys_vecs = [], []
				num_sam = int(self.sampling*len(self.l_vecs))
				for i_idx, xv in enumerate(x_vecs):
					cur_pos_ids = y_vecs[i_idx].nonzero()[0].tolist()
					all_ids = list(set(self.init_ids) - set(cur_pos_ids))
					sample_ids = np.random.randint(len(all_ids),  size=int(num_sam)).tolist()
					merged_ids = cur_pos_ids + np.array(all_ids)[sample_ids].tolist()
					samples = self.l_vecs[merged_ids][:num_sam]
					ls_vecs.append(samples)
					y_samples = y_vecs[i_idx][merged_ids][:num_sam]
					ys_vecs.append(y_samples)
				return [np.array(x_vecs), np.array(ls_vecs)], np.array(ys_vecs)
			else:
				return [np.array(x_vecs), self.bsl_vecs],  np.array(y_vecs)
		else:
			return np.array(x_vecs), np.array(y_vecs)


class MHAN:
	"""
	Class which contains all the necessary functions to create and train
	multilingual hierarchical attention neural networks based on three
	component sharing configurations:
		1. Sharing encoders at both levels
		2. Sharing attention at both levels
		3. Sharing encoders and attention at both levels
	"""

	def __init__(self, args):
		self.args = args
		self.revids = []
		self.single_language = len(self.args['languages']) == 1
		self.attention_mode = self.args['enc'].find('att') > -1
		self.args["languages"] = [args["source"]] if args["source"] else args["languages"]

	def build_encoders(self):
		""" Builds functions needed for the word-level and sentence-level
			encoders and returns them in a dictionary. """
		backsent_enc, backdoc_enc = None, None
		if self.args['enc'] in ["dense","attdense"]:
			sent_enc = TimeDistributed(Dense(self.args['sdim'],
							input_shape=(self.args['wpad'], self.args['wdim']),
							activation=self.args['act']),
							input_shape=(self.args['wpad'], self.args['wdim']))
			doc_enc = TimeDistributed(Dense(self.args['ddim'],
							input_shape=(self.args['spad'], self.args['wdim']),
							activation=self.args['act']),
							input_shape=(self.args['spad'], self.args['wdim']))
		elif self.args['enc'] in ["gru", "attgru"]:
			sent_enc = GRU(self.args['sdim'],
					input_shape=(self.args['wpad'], self.args['wdim']),
					activation=self.args['gruact'],
					return_sequences=True)
			doc_enc = GRU(self.args['ddim'],
				      input_shape=(self.args['spad'], self.args['wdim']),
				      activation=self.args['gruact'],
				      return_sequences=True)
		elif self.args['enc'] in ["bigru", "attbigru"]:
			sent_enc = GRU(self.args['sdim'],
						   input_shape=(self.args['wpad'], self.args['wdim']),
						   activation=self.args['gruact'],
						   return_sequences=True)
			backsent_enc = GRU(self.args['sdim'],
					   input_shape=(self.args['wpad'], self.args['wdim']),
					   go_backwards=True,
					   activation=self.args['gruact'],
					   return_sequences=True)
			doc_enc = GRU(self.args['ddim'],
				      input_shape=(self.args['spad'], self.args['wdim']*2),
				      activation=self.args['gruact'],
				      return_sequences=True)
			backdoc_enc = GRU(self.args['ddim'],
					  input_shape=(self.args['spad'], self.args['wdim']*2),
					  go_backwards=True,
					  activation=self.args['gruact'],
					  return_sequences=True)
		encoders = {'sent_enc': sent_enc,
			    'doc_enc': doc_enc,
			    'backsent_enc': backsent_enc,
			    'backdoc_enc': backdoc_enc}
		return encoders

	def build_attention(self, lang):
		""" Builds functions needed for the word-level and sentence-level
		    attention mechanisms. """
		bigru = self.args['enc'].find('bigru') > -1
		hsdim = self.args['sdim']*2 if bigru else self.args['sdim']
		hddim = self.args['ddim']*2 if bigru else self.args['ddim']
		sent_enc = TimeDistributed(Dense(hddim, activation=self.args['act']))
		sent_context = TimeDistributed(Dense(1, activation=self.args['act']))
		word_enc = TimeDistributed(Dense(hsdim, activation=self.args['act']))
		word_context = TimeDistributed(Dense(1, activation=self.args['act']))
		submax_sent = Lambda(self.submax, output_shape=self.submax_output)
		submax_word = Lambda(self.submax, output_shape=self.submax_output)
		softmax_sent = Activation(activation="softmax", name="%s_satt" % lang)
		softmax_word = Activation(activation="softmax", name="%s_watt" % lang)
		reshape_word = Reshape((self.args['spad'],self.args['swpad']))
		repeat_word = RepeatVector(hsdim)
		repeat_sent = RepeatVector(hddim)
		permute_sent = Permute((2,1))
		permute_word = Permute((2,1))
		flatten_sent = Flatten()
		flatten_word = Flatten()
		flatten_word_after = Flatten()
		attention = {'sent_enc': sent_enc,
			     'sent_context': sent_context,
			     'word_enc': word_enc,
			     'word_context': word_context,
			     'flatten_sent': flatten_sent,
			     'flatten_word': flatten_word,
			     'flatten_word_after': flatten_word_after,
			     'submax_sent': submax_sent,
			     'submax_word': submax_word,
			     'softmax_sent': softmax_sent,
			     'softmax_word': softmax_word,
			     'repeat_sent': repeat_sent,
			     'repeat_word': repeat_word,
			     'permute_sent': permute_sent,
			     'permute_word': permute_word,
			     'reshape_word': reshape_word}
		return attention

	def word_attention(self, forward_words, attention):
		""" Compute word-level attention scores and return attented
		    word vectors for the whole word sequence. """
		hdim = forward_words._keras_shape[2]
		embedded_words = attention['word_enc'](forward_words)
		attented_words = attention['word_context'](embedded_words)
		submaxed = attention['submax_word'](attented_words)
		weights = attention['softmax_word'](submaxed)
		return Multiply()([weights, forward_words])

	def wordpool(self, encoded_words):
		""" Compose a sentence representation given the encoded word
		    vectors in the given word sequence. """
		return K.sum(encoded_words, axis=1)

	def wordpool_output(self, input_shape):
		""" Defines the dimensions of the resulting sentence vector. """
		return tuple([None, input_shape[2]])

 	def sentencepool(self, encoded_sentences):
		""" Compose a document representation given the encoded sentence
		 	vectors in the given sentence sequence. """
		if not self.attention_mode:
			return K.mean(encoded_sentences, axis=1)
		return K.sum(encoded_sentences, axis=1)

	def sentencepool_output(self, input_shape):
		""" Defines the dimensions of the resulting document vector. """
		return tuple([None, input_shape[2]])

	def submax(self, x):
		""" Subtracts from each vector the value of its dimension with
		    the maximal value. """
		return x - K.max(x, axis=-1, keepdims=True)

	def submax_output(self, input_shape):
		""" Defines the dimensions of the output of the submax function. """
		return tuple(input_shape)

	def lcompose(self, x):
		""" Compose a label representation given the word vectors in
		    the given word sequence (description). """
		if self.args['maskedavg']:
			mask = K.not_equal(K.sum(K.abs(x), axis=3, keepdims=False), 0)
			n = K.sum(K.cast(mask, 'float32'), axis=2, keepdims=True)
			x_mean = K.sum(x, axis=2, keepdims=False) / n
		else:
			x_mean = K.mean(x, axis=2)
		return x_mean

	def lcompose_output(self, input_shape):
		""" Defines the dimensions of the resulting label vectors. """
		return tuple([None, input_shape[1], input_shape[3]])

	def build_joint(self, L=None):
		""" Initialize the joint input-output embedding space """
		default_dim = (self.args['ddim']*L)/(self.args['ddim']+self.args['wdim'])
		l = default_dim if self.args['ladim'] is None else self.args['ladim']
		classdoc_emb = Dense(l, input_dim=(self.args['ddim']), activation=self.args['laact'])
		classlab_emb = Dense(l, input_dim=(self.args['wdim'],), activation=self.args['laact'])
		joint = {'classdoc_emb': classdoc_emb,
				 'classlab_emb': classlab_emb,}
		return joint

	def build_model(self, encoders, attention, num_labels, joint=None):
		""" Builds a hierarchical attention eural network model based
		    on a given set of encoders and attention mechanisms. """
		input_model = Sequential()
		if self.args['enc'] in ["bigru", "attbigru"]:
			words = Input(shape=(self.args['wpad'],self.args['wdim'],))
			forward_words = encoders['sent_enc'](words)
			backward_words = encoders['backsent_enc'](words)
			bigru_words = Concatenate()([forward_words, backward_words])
			if self.args['enc'] == "attbigru":
				bigru_words = self.word_attention(bigru_words, attention)
			word_pooling = Lambda(self.wordpool, output_shape=self.wordpool_output)
			sentences = word_pooling(bigru_words)
			forward_sentences = encoders['doc_enc'](sentences)
			backward_sentences = encoders['backdoc_enc'](sentences)
			bigru_sentences = Concatenate()([forward_sentences, backward_sentences])
			if self.args['enc'] == "attbigru":
				bigru_sentences = self.sentence_attention(bigru_sentences, attention)
			sentence_pooling = Lambda(self.sentencepool, output_shape=self.sentencepool_output)
			document = sentence_pooling(bigru_sentences)
		elif self.args['enc'] in ["dense", "attdense", "gru", "attgru"]:
			words = Input(shape=(self.args['wpad'],))
			if self.args['pretrained']:
				wvec, vocab = load_word_vectors(self.args['languages'][0], self.args['wordemb_path']+'%s.pkl' % self.args['languages'][0])
				embed = Embedding(self.args['wsize'], self.args['wdim'],  weights=[np.array(wvec['english'])], trainable=False)
			else:
				embed = Embedding(self.args['wsize'], self.args['wdim'], trainable=True)
			forward_words = encoders['sent_enc'](embed(words))
			attented_words  = self.word_attention(forward_words, attention)
			word_pooling = Lambda(self.wordpool, output_shape=self.wordpool_output)
			document = word_pooling(attented_words)
		if self.args['la']:
			L = num_labels if self.args['test'] else int(self.args['sampling'] * num_labels)
			label_vecs = Input(shape=(L, self.args['lpad']))
			label_seqs = embed(label_vecs)
			lcompose = Lambda(self.lcompose, output_shape=self.lcompose_output)
			labels = lcompose(label_seqs)

			if self.args['onlylabel']:
				print "---> Using only label embedding."
				W_doc = document
				V_doc = joint['classlab_emb'](labels)
			elif self.args['onlyinput']:
				W_doc = joint['classdoc_emb'](document)
				V_doc = labels
				print "---> Using only input embedding."
			else:
				W_doc = joint['classdoc_emb'](document)
				V_doc = joint['classlab_emb'](labels)

			doclab_rep = RepeatVector(L)
			classjoint_sig =  Dense(1, input_dim=(V_doc._keras_shape[1]), activation='sigmoid')
			doclab_sig = TimeDistributed(classjoint_sig, )
			doclab_sig_reshape = Reshape((L,))
			W_rep = doclab_rep(W_doc)
			matrix = merge([W_rep, V_doc], "mul")
			decision = doclab_sig(matrix)
			decision = doclab_sig_reshape(decision)

			return words, label_vecs, decision
		else:
			classifier = Dense(num_labels, input_dim=(document._keras_shape[1]), activation='sigmoid')
			decision = classifier(document)
			return words, decision

	def get_inputs(self, inputs, input_labels):
		inputs_all = []
		for i in range(len(inputs)):
			inputs_all.append(inputs[i])
			inputs_all.append(input_labels[i])
		return inputs_all

	def build_multilingual_model(self, labels):
		""" Builds a multilingual hierarchical attention neural network
		    model based on a given component sharing configurations. """
		inputs, outputs, input_labels, joint = [], [], [], None
 		if self.args['share'].find('enc') > -1:
			encoders = self.build_encoders()
		elif self.args['share'].find('att') > -1:
			attention = self.build_attention(lang='both')
		elif self.args['share'].find('both') > -1:
			encoders = self.build_encoders()
			attention = self.build_attention(lang='both')
		if self.args['lashare']:
			joint = self.build_joint(L=sum([len(l) for l in labels]))
		for l, language in enumerate(self.args['languages']):
			if self.args['share'].find('enc') > -1:
				attention = self.build_attention(lang=language)
			elif self.args['share'].find('att') > -1:
				encoders = self.build_encoders()
			elif self.args['share'] == "none":
				encoders = self.build_encoders()
				attention = self.build_attention(lang=language)
			if self.args['la'] and not self.args['lashare']:
				joint = self.build_joint(L=len(labels[l]))
			if self.args['la']:
				words, label_vecs, preds = self.build_model(encoders, attention, len(labels[l]), joint=joint)
				input_labels.append(label_vecs)
			else:
				words, preds = self.build_model(encoders, attention, len(labels[l]))
			inputs.append(words)
			outputs.append(preds)
		if self.single_language:
			inputs, outputs = inputs[0], outputs[0]
			if self.args['la']:
				input_labels = input_labels[0]
		if self.args['la']:
			if self.single_language:
				input_model = Model(inputs=[inputs, input_labels], outputs=outputs)
			else:
				input_model = Model(input=self.get_inputs(inputs, input_labels), output=outputs)
		else:
			input_model = Model(inputs=inputs, outputs=outputs)
		input_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
		self.model = input_model
		return self.model

	def forward_attention(self):
		""" Define functions to get the attention scores at both levels. """
		self.watts, self.satts = [], []
		for l, language in enumerate(self.args['languages']):
			name_watt = "%s_watt" % language
			name_satt = "%s_satt" % language
			if len(self.args['languages']) > 1 and self.args['share'] != 'enc':
				name_watt = "both_watt"
				name_satt = "both_satt"
			if self.model.get_layer(name_watt) is not None:
				watt = theano.function([self.model.layers[l].input], self.model.get_layer(name_watt).get_output_at(l), allow_input_downcast=True)
				satt = theano.function([self.model.layers[l].input], self.model.get_layer(name_satt).get_output_at(l), allow_input_downcast=True)
				self.watts.append(watt)
				self.satts.append(satt)

	def load_vecs(self, wvec, labels):
		vecs = []
		keys = labels.keys()
		revids = np.zeros(len(keys), dtype=int)
		for j, lab in enumerate(keys):
			lids = labels[lab].value
			ids = np.zeros(self.args['lpad'])
			if len(lids) > self.args['lpad']:
				ids[:self.args['lpad']] = lids[:self.args['lpad']]
			else:
				ids[:len(lids)] = lids
			vecs.append(ids)
			revids[int(lab)] = j
		return vecs, np.array(revids)


	def fit(self, X_train, Y_train, X_val, Y_val, labels, wvecs, vocabs):
		""" Trains the model using stochastic gradient descent. At each epoch
			epoch, it stores the parameters of the model and its performance
			on the validation set. """
		resume_path, resume_epoch = self.find_checkpoint()
		errors, prs, recs, fs = [], [], [], []
		val_scores, train_scores = [], []
		label_vecs, revids = self.load_vecs(wvecs, labels[0])
		self.revids = revids
		params = {
		'X': X_train[0],
		'Y': Y_train[0],
		'l_vecs': label_vecs,
		'wpad': self.args['wpad'],
		'bs': self.args['bs'],
		'sampling': self.args['sampling'],
		'model': self
		}
		self.model.fit_generator(generator=DataGenerator(**params).generate(np.arange(self.args['ep_size'])),
					  			 steps_per_epoch=self.args['ep_size']/self.args['bs'],
								 epochs=self.args['ep'], verbose=1,
								 callbacks=[ValidateCallback(X_val[0], Y_val[0], label_vecs, self)],
								 initial_epoch=resume_epoch)

	def find_checkpoint(self):
		""" Check if there is a stored model to resume from. """
		try:
			path = "%s/" % (self.args['path'])
			fnames = [f for f in os.listdir(path) if f.find('weights') > -1]
			cur_e = np.sort([int(f.split('_')[1].split('-')[0]) for f in fnames])[-1]
			cur_idx = np.argsort([int(f.split('_')[1].split('-')[0]) for f in fnames])[-1]
			resume_path = path + fnames[cur_idx]
			self.model.load_weights(resume_path)
			print "[*] Resuming from epoch %d... (%s)" % (cur_e+1, resume_path)
			return resume_path, cur_e + 1
		except:
			print "[*] No stored model to resume from. "
			return None, 0

	def get_avgresults(self, preds, Y_vecs):
		""" Return the average precision, recall and f-measure computed
			over all languages. """
		pr, rec, f, = [], [], []
		if self.single_language:
			preds = [preds]
		for l, pred in enumerate(preds):
			pred = pred>self.args['t']
			prf = self.get_results(Y_vecs[l],  pred, print_result=False)
			pr.append(prf[0]);rec.append(prf[1]);f.append(prf[2])
		return sum(pr)/len(pr), sum(rec)/len(rec), sum(f)/len(f)

	def save_model(self, language, epoch, train_score, val_score):
		""" Store model and validation score at each epoch. """
		name = "epoch_%d" % epoch
		path = "%s/" % (self.args['path'])
		if not os.path.exists(path):
			os.makedirs(path)
		json_string = self.model.to_json()
		open(path+self.args['args_file'], 'w').write(json.dumps(self.args, indent=4))
		self.model.save_weights(path+'%s-weights.h5' % name)
		open(path+'%s-val.txt' % name,'w').write(' '.join([str(v) for v in val_score]))


	def eval(self, cur_lang, x, y, label_vecs, bs=8, av='micro', L=0, source=None, avg=True, mode='none'):
		""" Evaluate model on the given validation or test set. """
		cur_lang = 0 if source is not None else cur_lang
		preds, real, watts, satts = [], [], [], []
		batch, elapsed, curbatch, init = 0, 0, 0, 0
		rls, aps, oes, elapsed = [], [], [], 0.0
		total = len(x)
		keys = x.keys()
		num_labels = label_vecs.shape[1]
		if mode and mode == 'seen':
			eval_ids = pickle.load(open(self.args['seen_ids']))
			eval_ids = self.revids[eval_ids] # select evaluation ids
		elif mode and mode == 'unseen':
			eval_ids = pickle.load(open(self.args['unseen_ids']))
			eval_ids = self.revids[eval_ids] # select evaluation ids
		else: # validation
			eval_ids = np.arange(label_vecs.shape[1])
			total = 5000 # use a small sample for validation / otherwise too slow

		print
		while batch < total/(1.0*bs):
			start_time = time.time()
			init_ids = [init+curbatch+cur for cur in range(bs) if init+curbatch+cur < len(keys)]
			idxs = np.array(keys)[init_ids]
			x_vecs, y_vecs = load_vectors(x, y, idxs, self.args['wpad'], num_labels, self)
			if self.args['la']: # Zero-shot models
				if self.args['train']:
                    # Predictions for all the labels are build subsequently due to
					# the predefined vocabulary size which is required by sampling.
					ll = int(self.args["sampling"]*num_labels)
					done, pred, pi = False, None, 0
					while (not done):
						if pi == 0:
							totest = label_vecs[:,:ll]
						elif pi > 0:
							totest = label_vecs[:,pi*ll:ll+pi*ll]
							if totest.shape[1] != ll:
								remained = totest.shape[1]
								totest = np.hstack([totest, np.zeros((bs,ll - totest.shape[1], totest.shape[2]))])
								done = True
						cur_pred = self.model.predict([np.array(x_vecs), totest], batch_size=self.args['bs'])
						if pred is None:
							pred = cur_pred
						else:
							if done:
								pred = np.hstack([pred, cur_pred[:,:remained]])
							else:
								pred = np.hstack([pred, cur_pred])
						pi += 1
				else:
					pred = self.model.predict([np.array(x_vecs), label_vecs], batch_size=self.args['bs'])
			else:
                # Non-zero-shot models
				pred = self.model.predict(np.array(x_vecs), batch_size=self.args['bs'])
			real = np.array(y_vecs); pred = np.array(pred)
			rls.append(rankloss(real[:,eval_ids], pred[:,eval_ids]))
			aps.append(avgprec(real[:,eval_ids], pred[:,eval_ids]))
			cur_oes = [one_error(real[j][eval_ids], pred[j][eval_ids]) for j in range(len(pred))]
			oes.append(np.array(cur_oes).mean())
			elapsed += time.time() - start_time
 			sys.stdout.write("\t%d/%d rls=%.5f - aps=%.5f - oe=%.5f \t %ds\r"%(((batch+1)*bs), len(x),
                             np.array(rls).mean(), np.array(aps).mean(), np.array(oes).mean(), elapsed))
			sys.stdout.flush()
			batch += 1; curbatch += bs
		if avg:
			rls = np.array(rls).mean()
			aps = np.array(aps).mean()
			oes = np.array(oes).mean()
			print "rl: %.4f - ap: %.4f - oe: %.4f" % (rls, aps, oes)
	 		return rls, aps, oes
		else:
			return rls, aps, oes

	def get_results(self, reals, preds, av="micro", print_result=True):
		""" Calculates and prints precision, recall and f-measure based
		 	on the real and predicted categories. """
		prf = precision_recall_fscore_support(reals, preds, average=av)
		if print_result:
			print "\t**val p: %.5f - r: %.5f - f: %.5f " % (prf[0], prf[1], prf[2])
		return [ prf[0], prf[1], prf[2] ]
