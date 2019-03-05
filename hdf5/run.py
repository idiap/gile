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


import os, sys, h5py
import numpy as np
import argparse, json, keras
from models import MHAN
if keras.__version__[0] == "1":
    from keras.utils.visualize_util import plot as plot_model
else:
    from keras.utils import plot_model
from util import load_data, load_word_vectors, pick_best, export, load_missing_args


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='GILE toolkit with HDF5 support (based on http://github.com/idiap/mhan)')
	parser.add_argument('--wdim', type=int, default=40, help='Number of dimensions of the word embeddings. ')
	parser.add_argument('--wsize', type=int, default=528139, help='Number of words in the vocabulary of the word embeddings. ')
	parser.add_argument('--lpad', type=int, default=50, help='Maximum number of words in a label text.')
	parser.add_argument('--wpad', type=int, default=300, help='Maximum number of words in a text.')
	parser.add_argument('--sampling', type=float, default=1.0, help='Percentage of labels to train on.')
	parser.add_argument('--swpad', type=int, default=30, help='Maximum number of words in a sentence.')
	parser.add_argument('--spad', type=int, default=30, help='Maximum number of sentences in a document.')
	parser.add_argument('--sdim', type=int, default=100, help='Number of hidden dimensions of the word-level encoder.')
	parser.add_argument('--ddim', type=int, default=100, help='Number of hidden dimensions of the sentence-level encoder.')
	parser.add_argument('--ep', type=int, default=200, help='Maximum number of epochs for training.')
	parser.add_argument('--ep_size', type=int, default=6692815, help='Maximum number of samples per epoch during training.')
	parser.add_argument('--epval_size', type=int, default=100000, help='Maximum number of samples per epoch during training.')
	parser.add_argument('--bs', type=int, default=16, help='Size of batch to be used for training.')
	parser.add_argument('--enc', type=str, default='attdense', help='Type of encoder and pooling layer to be used at each level of the hierarchical model, namely dense, gru or bigru (transformation + average pooling) or attdense, attgru, attbigru (transformation + attention mechanism).')
	parser.add_argument('--act', type=str, default='relu', help='Activation to be used for the Dense layers.')
	parser.add_argument('--gruact', type=str, default='tanh', help='Activation to be used for the GRU/biGRU layers.')
	parser.add_argument('--share', type=str, default='none', help='Component to share in the multilingual model: encoders (enc), attention (att) or both (both).')
	parser.add_argument('--t', type=float, default=0.20, help='Decision threshold to be used for validation or testing.')
	parser.add_argument('--seed', type=int, default=1234, help='Random seed number.')
	parser.add_argument('--args_file', type=str,  default='args.json', help='Name of the file in the experiment folder where the settings of the model are stored.')
	parser.add_argument('--wordemb_path', type=str,  default='word_vectors/', help='<Required:train|test|store_test> Path of the word vectors in pickle format for each language (e.g. english.pkl, etc).')
	parser.add_argument('--languages', nargs='+', default=None, help='<Required:train> Languages to be used for multilingual training.')
	parser.add_argument('--data_path', type=str,  default='data/bioasq', help='<Required:train|test|store_test> Path of the train/, dev/ and test/ folders which contain data in json format for each language.')
	parser.add_argument('--path', type=str,  default='exp/english', help='<Required:train> Path of the experiment in which the model parameters and validation scores are stored after each epoch.')
	parser.add_argument('--target', type=str,  default=None, help='<Required:test> Language in which the testing should be performed.')
	parser.add_argument('--source', type=str,  default=None, help='<Optional:test> Language in which model should be loaded from. Useful only for cross-lingual tagging (user with --store_test option).')
	parser.add_argument('--store_file', type=str, default='results.json', help='<Optional:store_test> Name of the file in the experiment folder where the predictions and attention scores of the model are stored.')
	parser.add_argument('--chunks', type=int, default=50, help='<Optional:train> Number of chunks for the data files.')
	parser.add_argument('--train', action='store_true', help='Train the model from scratch.')
	parser.add_argument('--test', action='store_true', help='Test the model.')
	parser.add_argument('--store_test', action='store_true', help='Store the predictions and attention scores of the model on the test set.')
	parser.add_argument('--mode', default=None, help='Evaluation mode (seen | unseen).')
	parser.add_argument('--pretrained', action='store_true', help='Use pre-trained word embeddings.')
	parser.add_argument('--maskedavg', action='store_true', help='Use masked averaging for label vectors.')
	parser.add_argument('--chunks_offset', type=int, default=0, help='Chunks offset.')
	parser.add_argument('--onlylabel', action='store_true', help='Use only a label embedding.')
	parser.add_argument('--onlyinput', action='store_true', help='Use only an input embedding.')
	parser.add_argument('--la', action='store_true', help='Use the generalized input-output embedding output layer.')
	parser.add_argument('--ladim', type=int, default=None, help='Number of hidden dimensions of the generalized input-label embedding.')
	parser.add_argument('--laact', type=str, default='relu', help='Activation to be used for the generalized input-label projection.')
	parser.add_argument('--lashare', default=None, action='store_true', help='Share the generalized embedding across languages or not.')
	parser.add_argument('--seen_ids', type=str, default='data/bioasq/test/seen_ids.pkl', help='Path to seen ids during training stored in Pickle format.')
	parser.add_argument('--unseen_ids', type=str, default='data/bioasq/test/unseen_ids.pkl', help='Path to unseen ids during training stored in Pickle format.')


	X_ids, Y_ids, XV_ids, YV_ids, XT_ids = [], [], [], [], []
	YT_ids, wvecs, labels, vocabs = [], [], [], []
	parsed_args = parser.parse_args()

	try:
		json_string = open("%s/%s" % ( parsed_args.path, parsed_args.args_file) ).read()
		args = json.loads(json_string)
		args, parsed_args = load_missing_args(args, parsed_args)
	except:
		args = parsed_args.__dict__

	for language in args['languages']:
		wordemb_path = args['wordemb_path']+'%s.pkl' %  language
		wvec, vocab = [], []
		if parsed_args.train:
			x_ids, y_ids = [], []
			for chunk_id in range(args["chunks"]):
				train_path = args['data_path']+'/train/%s-%d.h5' % (language, chunk_id)
				x, y, c = load_data(path=train_path)
				x_ids.append(x); y_ids.append(y)
			dev_path = args['data_path']+'/dev/%s.h5' %  language
			xv_ids, yv_ids, cur_labels = load_data( path=dev_path)
			print "\tX_train (80%)"+": %d" % len(x_ids)
			print "\tX_val (20%)"+": %d" % len(xv_ids)
			X_ids.append(x_ids);Y_ids.append(y_ids)
			XV_ids.append(xv_ids);YV_ids.append(yv_ids)
		elif parsed_args.test or parsed_args.store_test:
			xt_ids, yt_ids = [], []
			for chunk_id in range(args["chunks"]):
				if parsed_args.mode == 'unseen':
					test_path = args['data_path']+'/unseen_test/%s-%d.h5' % (language, chunk_id +
						parsed_args.chunks_offset)
				else:
					test_path = args['data_path']+'/test/%s-%d.h5' % (language, chunk_id +
						parsed_args.chunks_offset)
				x, y, c = load_data(path=test_path)
				xt_ids.append(x); yt_ids.append(y)

				dev_path = args['data_path']+'/dev/%s.h5' %  language
				xv_ids, yv_ids, cur_labels = load_data( path=dev_path)

			print "\tX_test (10%)"+": %d" % len(xt_ids)
			XT_ids.append(xt_ids);YT_ids.append(yt_ids)

		print "\t|V|: %d, |Y|: %d" % (args['wsize'], len(cur_labels))
		labels.append(cur_labels)

	mhan = MHAN(args)
	mhan.build_multilingual_model(labels)

	if parsed_args.train:
		print "[*] Training model..."
		mhan.fit(X_ids, Y_ids, XV_ids, YV_ids, labels, wvecs, vocabs)
	if parsed_args.test or parsed_args.store_test:
		lang_idx = parsed_args.languages.index(args['target'])
		dev_path = "%s/" % ( args['path'])
		source_idx = lang_idx
		if parsed_args.source is not None:
			print "[*] Cross-lingual mode: ON"
			print "[*] Source language: %s" % args['source']
			dev_path = "%s/%s/" % ( args['path'], args['source'])
			source_idx = parsed_args.languages.index(args['source'])
		epoch_num, best_weights_file = pick_best(dev_path)
		mhan.model.load_weights(best_weights_file)
		plot_model(mhan.model, to_file="%sarch.png" % dev_path)

		print "[*] Testing model on %s..." % args['target']

		label_vecs, revids = mhan.load_vecs(wvecs, labels[0])
		mhan.revids = revids
		bsl_vecs = np.array([label_vecs for j in range(parsed_args.bs)])
		rls, aps, oes = [], [], []

		for j, xt_ids in enumerate(XT_ids[lang_idx]):
			cur_rls, cur_aps, cur_oes = mhan.eval(lang_idx, XT_ids[lang_idx][j],
                                                 YT_ids[lang_idx][j], bsl_vecs,
                                                 L=len(parsed_args.languages),
                                                 source=parsed_args.source,
                                                 avg=False, mode=parsed_args.mode,
                                                 bs=parsed_args.bs)
			rls += cur_rls; aps += cur_aps; oes += cur_oes

		res = [np.array(rls).mean(), np.array(aps).mean(), np.array(oes).mean()]
		print "\n"
		print res
 		out_file = "%sepoch_%d-test-%s-%d.txt" % (dev_path, epoch_num,
                                                 parsed_args.mode,
                                                 parsed_args.chunks_offset)
		open(out_file, 'w').write(' '.join([str(v) for v in res]))
	print "[-] Finished."
