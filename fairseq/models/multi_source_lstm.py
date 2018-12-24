# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models.lstm import LSTMEncoder, LSTMDecoder, Embedding, LSTM
from fairseq.modules import AdaptiveSoftmax
from . import (
    FairseqEncoder, FairseqIncrementalDecoder, FairseqModel, register_model,
    register_model_architecture,
)


@register_model('multi_source_lstm')
class MultiSourceLSTMModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
                            help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true',
                            help='make all layers of encoder bidirectional')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
                            help='decoder attention')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for encoder output')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        parser.add_argument('--share-decoder-input-output-embed', default=False,
                            action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError('--encoder-layers must match --decoder-layers')

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings_1 = len(task.source_dictionary[0])
            num_embeddings_2 = len(task.source_dictionary[1])
            if args.encoder_bidirectional:
                pretrained_encoder_embed = [
                    Embedding(num_embeddings_1, args.encoder_embed_dim, task.source_dictionary[0].pad()),
                    Embedding(num_embeddings_2, 2 * args.encoder_embed_dim, task.source_dictionary[1].pad()),
                ]
            else:
                pretrained_encoder_embed = [
                    Embedding(num_embeddings_1, args.encoder_embed_dim, task.source_dictionary[0].pad()),
                    Embedding(num_embeddings_2, args.encoder_embed_dim, task.source_dictionary[1].pad()),
                ]

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        encoder = MultiSourceLSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
        )
        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
        )
        return cls(encoder, decoder)


class MultiSourceLSTMEncoder(FairseqEncoder):
    """LSTM encoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        left_pad=True, pretrained_embed=None, padding_value=0.,
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        num_embeddings1 = len(dictionary[0])
        num_embeddings2 = len(dictionary[1])
        self.padding_idx_1 = dictionary[0].pad()
        self.padding_idx_2 = dictionary[1].pad()
        if pretrained_embed is None:
            self.embed_tokens_1 = Embedding(num_embeddings1, embed_dim, self.padding_idx_1)
            if bidirectional:
                self.embed_tokens_2 = Embedding(num_embeddings2, 2 * embed_dim, self.padding_idx_2)
            else:
                self.embed_tokens_2 = Embedding(num_embeddings2, embed_dim, self.padding_idx_2)
        else:
            self.embed_tokens_1, self.embed_tokens_2 = pretrained_embed

        self.lstm1 = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.lstm2 = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2


    def forward(self, src_tokens, src_lengths):
        src_tokens1, src_tokens2 = src_tokens
        src_lengths1, src_lengths2 = src_lengths
        if self.left_pad:
            # convert left-padding to right-padding
            src_tokens1 = utils.convert_padding_direction(
                src_tokens1,
                self.padding_idx_1,
                left_to_right=True,
            )
            src_tokens2 = utils.convert_padding_direction(
                src_tokens2,
                self.padding_idx_2,
                left_to_right=True,
            )

        bsz1, seqlen1 = src_tokens1.size()
        bsz2, seqlen2 = src_tokens2.size()

        # embed tokens
        x1 = self.embed_tokens_1(src_tokens1)
        x1 = F.dropout(x1, p=self.dropout_in, training=self.training)
        x2 = self.embed_tokens_2(src_tokens2)
        x2 = F.dropout(x2, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x1 = x1.transpose(0, 1)
        x2 = x2.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x1 = nn.utils.rnn.pack_padded_sequence(x1, src_lengths1.data.tolist())
        # packed_x2 = nn.utils.rnn.pack_padded_sequence(x2, src_lengths2.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size1 = 2 * self.num_layers, bsz1, self.hidden_size
            # state_size2 = 2 * self.num_layers, bsz2, self.hidden_size
        else:
            state_size1 = self.num_layers, bsz1, self.hidden_size
            # state_size2 = self.num_layers, bsz2, self.hidden_size
        h01 = x1.data.new(*state_size1).zero_()
        c01 = x1.data.new(*state_size1).zero_()
        packed_outs1, (final_hiddens1, final_cells1) = self.lstm1(packed_x1, (h01, c01))
        # h02 = x2.data.new(*state_size2).zero_()
        # c02 = x2.data.new(*state_size2).zero_()
        # packed_outs2, (final_hiddens2, final_cells2) = self.lstm2(packed_x2, (h02, c02))

        # unpack outputs and apply dropout
        x1, _ = nn.utils.rnn.pad_packed_sequence(packed_outs1, padding_value=self.padding_value)
        x1 = F.dropout(x1, p=self.dropout_out, training=self.training)
        assert list(x1.size()) == [seqlen1, bsz1, self.output_units]
        # x2, _ = nn.utils.rnn.pad_packed_sequence(packed_outs2, padding_value=self.padding_value)
        # x2 = F.dropout(x2, p=self.dropout_out, training=self.training)
        # assert list(x2.size()) == [seqlen2, bsz2, self.output_units]

        if self.bidirectional:

            def combine_bidir_1(outs):
                return outs.view(self.num_layers, 2, bsz1, -1).transpose(1, 2).contiguous().view(self.num_layers, bsz1, -1)
            # def combine_bidir_2(outs):
            #     return outs.view(self.num_layers, 2, bsz2, -1).transpose(1, 2).contiguous().view(self.num_layers, bsz2, -1)

            final_hiddens_1 = combine_bidir_1(final_hiddens1)
            final_cells_1 = combine_bidir_1(final_cells1)
            # final_hiddens_2 = combine_bidir_2(final_hiddens2)
            # final_cells_2 = combine_bidir_2(final_cells2)

        encoder_padding_mask_1 = src_tokens1.eq(self.padding_idx_1).t()
        encoder_padding_mask_2 = src_tokens2.eq(self.padding_idx_2).t()
        x = torch.cat([x1, x2])
        encoder_padding_mask = torch.cat([encoder_padding_mask_1, encoder_padding_mask_2])

        # HACK: pass hidden state of source 1 (title) to decoder 
        return {
            'encoder_out': (x, None, None),
            'encoder_out': (x, final_hiddens_1, final_cells_1),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


@register_model_architecture('multi_source_lstm', 'multi_source_lstm')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', args.encoder_embed_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', False)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_attention = getattr(args, 'decoder_attention', '1')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '10000,50000,200000')
