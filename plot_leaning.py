import matplotlib.pyplot as plt
LOG_FILE = 'log.train.txt'

if __name__ == '__main__':
    with open(LOG_FILE, 'r') as f:
        text = f.read()
        rez = text.split("START_LEANING\n")[-1]
        lines = [l for l in rez.split('\n')]
        
        train = [ float(line.split('train: ')[-1][:6]) for line in lines if len(line) > 5]
        val = [ float(line.split('val: ')[-1][:6]) for line in lines if len(line) > 5]
    plt.plot(train, label='train')
    plt.plot(val, label='val')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid()
    plt.show()