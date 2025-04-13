import argparse

import MNN
import whisper


class MNNModel:
    def __init__(self, mnn_model_path: str) -> None:
        input_names = ["audio"]
        output_names = ["embed"]

        self.net = MNN.nn.load_module_from_file(
            mnn_model_path,
            input_names,
            output_names,
        )

    def process(self, audio):
        mnn_audio = MNN.expr.placeholder(audio.shape, dtype=MNN.numpy.float32)

        mnn_audio.write(audio)

        output = self.net.forward([mnn_audio])
        output = MNN.expr.convert(output[0], MNN.expr.NCHW)
        embed = output.read().copy()
        return embed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mnn_model_path",
        required=True,
        help="path of mnn model path",
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
    model = MNNModel(
        mnn_model_path=args.mnn_model_path,
    )

    audio = whisper.load_audio(args.input_audio_path)[None]
    print(audio.shape)

    embed = model.process(audio)
    print("embedding's mean:", embed.mean())
