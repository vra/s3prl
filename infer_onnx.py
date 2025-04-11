import argparse

import onnxruntime
import whisper


class ONNXModel:
    def __init__(self, onnx_path):
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)

    def process(self, x):
        pred_audio = self.onnx_session.run(
            None,
            input_feed={
                "audio": x,
            },
        )
        embed = pred_audio[0]
        return embed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--onnx_path",
        required=True,
        help="path of onnx model path",
    )
    parser.add_argument(
        "-i",
        "--input_audio_path",
        required=True,
        help="path to wave file to infer",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model = ONNXModel(
        onnx_path=args.onnx_path,
    )

    audio = whisper.load_audio(args.input_audio_path)[None]

    embed = model.process(audio)
    print("embedding's mean:", embed.mean())
