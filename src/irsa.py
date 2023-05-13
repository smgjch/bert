import rsa


class Cipher():
    def __init__(self):
        self.public_key = rsa.PublicKey
        self.private_key = rsa.PrivateKey

    def write_keys(self):
        with open("./cipher/PublicKey.txt","w") as puk:
            puk.write(self.public_key.e)
        with open("./cipher/PrivateKey.txt","w") as puk:
            puk.write(self.private_key.e)

test = Cipher()
test.write_keys()