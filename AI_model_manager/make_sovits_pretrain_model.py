import torch


def clean_model(G, D):
    G_Model = torch.load(G)
    G_Model["iteration"] = 0
    G_Model["optimizer"] = None
    G_Model["learning_rate"] = 0.0001
    del G_Model["model"]["emb_g.weight"]
    torch.save(G_Model, "clean_G")

    D_Model = torch.load(D)
    D_Model["iteration"] = 0
    D_Model["optimizer"] = None
    D_Model["learning_rate"] = 0.0001
    torch.save(D_Model, "clean_D")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate sovits pre-trained Model")
    parser.add_argument("-g", "--g_model", required=True,
                        help="path to your G_Model")
    parser.add_argument("-d", "--d_model", required=True,
                        help="path to your D_Model")
    args = parser.parse_args()
    clean_model(args.g_model, args.d_model)
    print("done")
