from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices, Wav2Vec2PreTrainedModel, Wav2Vec2ForPreTraining
from transformers import Wav2Vec2Config, Wav2Vec2Model
from speechbrain.lobes import features
from torch.nn.utils.rnn import pad_sequence
