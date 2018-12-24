import itertools
import numpy as np
import os

from fairseq import options, utils
from fairseq.data import (
    data_utils, Dictionary, MultiSourceLanguagePairDataset, ConcatDataset,
    IndexedRawTextDataset, IndexedCachedDataset, IndexedDataset
)

from . import FairseqTask, register_task


@register_task('multi_source_translation')
class MultiSourceTranslation(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>` and :mod:`interactive.py <interactive>`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('-s1', '--source-lang1', default=None, metavar='SRC',
                            help='source language 1')
        parser.add_argument('-s2', '--source-lang2', default=None, metavar='SRC',
                            help='source language 2')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        # fmt: on

    # @staticmethod
    # def load_pretrained_model(path, src_dict_path, tgt_dict_path, arg_overrides=None):
    #     model = utils.load_checkpoint_to_cpu(path)
    #     args = model['args']
    #     state_dict = model['model']
    #     args = utils.override_model_args(args, arg_overrides)
    #     src_dict = Dictionary.load(src_dict_path)
    #     tgt_dict = Dictionary.load(tgt_dict_path)
    #     assert src_dict.pad() == tgt_dict.pad()
    #     assert src_dict.eos() == tgt_dict.eos()
    #     assert src_dict.unk() == tgt_dict.unk()

    #     task = MultiSourceTranslation(args, src_dict, tgt_dict)
    #     model = task.build_model(args)
    #     model.upgrade_state_dict(state_dict)
    #     model.load_state_dict(state_dict, strict=True)
    #     return model

    def __init__(self, args, src_dict1, src_dict2, tgt_dict):
        super().__init__(args)
        self.src_dict1 = src_dict1
        self.src_dict2 = src_dict2
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.source_lang = f"{args.source_lang1}_{args.source_lang2}"
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict1 = Dictionary.load(os.path.join(args.data[0], 'dict.{}.txt'.format(args.source_lang1)))
        src_dict2 = Dictionary.load(os.path.join(args.data[0], 'dict.{}.txt'.format(args.source_lang2)))
        tgt_dict = Dictionary.load(os.path.join(args.data[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict1.pad() == src_dict2.pad()
        assert src_dict1.eos() == src_dict2.eos()
        assert src_dict1.unk() == src_dict2.unk()
        assert src_dict1.pad() == tgt_dict.pad()
        assert src_dict1.eos() == tgt_dict.eos()
        assert src_dict1.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang1, len(src_dict1)))
        print('| [{}] dictionary: {} types'.format(args.source_lang2, len(src_dict2)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict1, src_dict2, tgt_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        src_datasets_1 = []
        src_datasets_2 = []
        tgt_datasets = []

        data_paths = self.args.data

        for dk, data_path in enumerate(data_paths):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                src1, src2 = self.args.source_lang1, self.args.source_lang2
                if split_exists(split_k, src, tgt, src1, data_path) and split_exists(split_k, src, tgt, src2, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src1, data_path) and split_exists(split_k, tgt, src, src2, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                src_datasets_1.append(indexed_dataset(prefix + src1, self.src_dict1))
                src_datasets_2.append(indexed_dataset(prefix + src2, self.src_dict2))
                tgt_datasets.append(indexed_dataset(prefix + tgt, self.tgt_dict))

                print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets_1[-1])))

                if not combine:
                    break

        assert len(src_datasets_1) == len(tgt_datasets)
        assert len(src_datasets_2) == len(tgt_datasets)

        if len(src_datasets_1) == 1:
            src_dataset_1, tgt_dataset = src_datasets_1[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets_1)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset_1 = ConcatDataset(src_datasets_1, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        if len(src_datasets_2) == 1:
            src_dataset_2, tgt_dataset = src_datasets_2[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets_2)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset_2 = ConcatDataset(src_datasets_2, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        self.datasets[split] = MultiSourceLanguagePairDataset(
            [src_dataset_1, src_dataset_2], [src_dataset_1.sizes, src_dataset_2.sizes], [self.src_dict1, self.src_dict2],
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return [self.src_dict1, self.src_dict2]

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
