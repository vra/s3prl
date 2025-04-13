import argparse

import numpy as np
import torch
import whisper


from demo import VoiceEmbeddingExtractor
from infer_onnx import ONNXModel
from infer_mnn import MNNModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--type",
        choices=["pytorch", "onnx", "mnn"],
        required=True,
        help="type of mode",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        required=True,
        help="path of pytorch/onnx/mnn model path",
    )
    parser.add_argument(
        "--input_audio_path1",
        required=True,
        help="path to wave file to infer",
    )
    parser.add_argument(
        "--input_audio_path2",
        required=True,
        help="path to wave file to infer",
    )

    args = parser.parse_args()
    return args


def verification(model, audio_path1, audio_path2):
    if isinstance(model, VoiceEmbeddingExtractor):
        func_name = "extract_embedding"
        audio1 = audio_path1
        audio2 = audio_path2
    else:
        func_name = "process"
        audio1 = whisper.load_audio(audio_path1)[None]
        audio2 = whisper.load_audio(audio_path2)[None]
    func = getattr(model, func_name)

    embed1 = func(audio1)
    embed2 = func(audio2)

    if isinstance(embed1, np.ndarray):
        embed1 = torch.from_numpy(embed1)
    if isinstance(embed2, np.ndarray):
        embed2 = torch.from_numpy(embed2)

    sim = torch.nn.functional.cosine_similarity(embed1, embed2).item()
    print(f"cosine_similarity: {sim:.3f}")


if __name__ == "__main__":
    args = parse_args()

    model_type = args.type
    model_path = args.model_path

    if model_type == "pytorch":
        model = VoiceEmbeddingExtractor(ckpt_path=model_path)
    elif model_type == "onnx":
        model = ONNXModel(onnx_path=model_path)
    elif model_type == "mnn":
        model = MNNModel(mnn_model_path=model_path)
    else:
        raise ValueError()

    verification(
        model=model,
        audio_path1=args.input_audio_path1,
        audio_path2=args.input_audio_path2,
    )
