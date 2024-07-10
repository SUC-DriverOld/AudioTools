import torch


def process_model(model_path, save_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    for i, j in model["model"].items():
        model["model"][i] = j.to(torch.float16)

    del model["optimizer"]
    torch.save(model, save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate RVC pre-trained model.")
    parser.add_argument('-g', '--g_model', required=True,
                        help="path to your G_Model.")
    parser.add_argument('-d', '--d_model', required=True,
                        help="path to your D_Model.")

    args = parser.parse_args()

    process_model(args.g_model, 'G_40k.pth')
    process_model(args.d_model, 'D_40k.pth')

    print("done")
