class S:
    def __init__(self):
        self.samira = "samira"

    def __getattr__(self, item):
        return str(item)

    def __getstate__(self):
        return "asd"

s = S
print(s())
