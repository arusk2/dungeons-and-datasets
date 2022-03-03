# This is a script for generating dungeons. Needs the trained weights from our model

if __name__ == '__main__':
    name =  "______ _   _ _____  _   _           ___ _____ ___________ \n" \
            "|  _  \ \ | |  __ \| \ | |         / _ \_   _|  _  | ___ \ \n" \
            "| | | |  \| | |  \/|  \| | ______ / /_\ \| | | | | | |_/ / \n" \
            "| | | | . ` | | __ | . ` | ______ |  _  || | | | | |    / \n" \
            "| |/ /| |\  | |_\ \| |\  |        | | | || | \ \_/ / |\ \ \n" \
            "|___/ \_| \_/\____/\_| \_/        \_| |_/\_/  \___/\_| \_| \n\n"

    print(name)
    print("Initializing oracle...")
    # model weights loaded here


    seed = input("Enter what you need... \n")
    print(seed)

    # Load model weights
    # build model
    # use seed as a the predictor
    # print numbered outputs

