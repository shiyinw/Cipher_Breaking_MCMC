import numpy as np
import math, random, time, collections
random.seed(121)
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool

P = np.zeros(shape=([28]))
with open("data/letter_probabilities.csv", "r") as f:
    line = f.readline()
    P = [float(x) for x in line[:-1].split(",")]
logP = np.log(P)


M = np.zeros(shape=([28, 28]))
with open("data/letter_transition_matrix.csv", "r") as f:
    lines = f.readlines()
    assert len(lines)==28, "The size of the alphabet is 28"
    for i in range(28):
        M[i, :] = [float(x) for x in lines[i].split(",")]
logM = np.log(M) # 0 exists


with open("data/alphabet.csv", "r") as f:
    line = f.readline()
    assert len(lines)==28, "The size of the alphabet is 28"
    Alphabet = line[:-1].split(",")
    content2idx = {}
    idx2content = {}
    for i in range(28):
        content2idx[Alphabet[i]] = i
        idx2content[i] = Alphabet[i]


class MCMC:
    def __init__(self, ciphertext):
        self.alphabet = Alphabet
        self.idx2alpha = dict(zip(range(28), self.alphabet))
        random.shuffle(self.alphabet)
        self.cur_f = dict(zip(self.alphabet, range(28)))
        self.ciphertext_transition = collections.Counter()
        for a in self.alphabet:
            self.ciphertext_transition[a] = collections.Counter()
        for i in range(1, len(ciphertext), 1): #including /n
            self.ciphertext_transition[ciphertext[i]][ciphertext[i-1]] += 1
        self.ciphertext = ciphertext
        self.set1, self.set2 = self.checkvalid(self.ciphertext)
        
    def Pf(self, code2idx):
        logPf = logP[code2idx[self.ciphertext[0]]]
        for a in self.alphabet:
            for b in self.alphabet:
                if self.ciphertext_transition[a][b]!=0:
                    if logM[code2idx[a], code2idx[b]] == -float('inf'):
                        return "not exist"
                    else:
                        logPf += self.ciphertext_transition[a][b] * logM[code2idx[a], code2idx[b]]
        
        return logPf

    def checkvalid(self, content):
        transition = collections.Counter()
        for i in self.alphabet:
            transition[i] = collections.Counter()
        for i in range(1, len(content), 1):  # including /n
            transition[content[i - 1]][content[i]] += 1

        valid = False

        notrepeat = []  # ' ' & '.'
        for a in self.alphabet:
            if (transition[a][a] == 0):
                notrepeat.append(a)

        set1 = []  # " "
        set2 = []  # "."
        for a in notrepeat:
            if (len(transition[a]) == 0):
                set2.append(a)
            elif (len(transition[a]) == 1 and list(transition[a].keys())[0] in notrepeat):
                set1.append(list(transition[a].keys())[0])
                set2.append(a)

        if (len(transition) == 28):  # All the char have occurred. So we must have a ". " pair.
            return set2, set1
        else:
            return set2, notrepeat

    def generate_f(self, oldf, set1, set2):
        # set1: "."
        # set2 : " "
        a, b = random.sample(self.alphabet, 2)
        f2 = oldf.copy()
        f2[a] = self.cur_f[b]
        f2[b] = self.cur_f[a]
        for i, v in f2.items():
            if (v == 27 and i not in set1):
                candidate = random.sample(self.set1, k=1)[0]
                tmp = f2[i]
                f2[i] = f2[candidate]
                f2[candidate] = tmp
            if (v == 26 and i not in set2):
                candidate = random.sample(self.set2, k=1)[0]
                tmp = f2[i]
                f2[i] = f2[candidate]
                f2[candidate] = tmp
        return f2

    
    def decode(self):
        s = ""
        for c in self.ciphertext:
            s += self.idx2alpha[self.cur_f[c]]
        return s
        
    
    def run(self, runningtime=30):
        start_time = time.time()
        loglikelihood = []
        accepted = []
        while(time.time()-start_time<runningtime):
            f2 = self.generate_f(self.cur_f, self.set1, self.set2)
            pf2 = self.Pf(f2)
            pf1 = self.Pf(self.cur_f)
            rand = random.random()
            if pf1=="not exist" and pf2=="not exist":
                if random.random()<0.5:
                    accepted.append(True)
                    self.cur_f= f2
                else:
                    accepted.append(False)
            elif pf1=="not exist":
                accepted.append(True)
                self.cur_f = f2
            elif pf2=="not exist":
                accepted.append(False)
                pass
            
            elif rand<min(1, np.exp(pf2-pf1)):
                self.cur_f = f2
                accepted.append(True)
                
            else:
                accepted.append(False)
            if self.Pf(self.cur_f) != "not exist":
                loglikelihood.append(self.Pf(self.cur_f))
            else:
                loglikelihood.append(float('nan'))
           
        return loglikelihood[-1], self.cur_f


class MCMC_B(MCMC):
    def __init__(self, ciphertext):
        MCMC.__init__(self, ciphertext)
        random.shuffle(self.alphabet)
        self.cur_f1 = dict(zip(self.alphabet, range(28)))
        random.shuffle(self.alphabet)
        self.cur_f2 = dict(zip(self.alphabet, range(28)))
        self.breakpoint = int(len(ciphertext)/2)

    def Pf(self, code2idx, breakpoint):
        logPf1 = logP[code2idx[ciphertext[breakpoint-1]]]
        for idx in range(1, breakpoint, 1):
            if logM[code2idx[idx], code2idx[idx-1]] == -float('inf'):
                return "not exist"
            else:
                logPf1 += logM[code2idx[a], code2idx[b]]

        logPf1 = logP[code2idx[ciphertext[breakpoint]]]
        for idx in range(breakpoint+1, len(self.ciphertext), 1):
            if logM[code2idx[idx], code2idx[idx-1]] == -float('inf'):
                return "not exist"
            else:
                logPf1 += logM[code2idx[a], code2idx[b]]
        return logPf1*logPf2

    def generate_f(self):
        a, b = random.sample(self.alphabet, 2)
        f2 = self.cur_f.copy()
        f2[a] = self.cur_f[b]
        f2[b] = self.cur_f[a]
        return f2

    def decode(self):
        s = ""
        for c in self.ciphertext:
            s += self.idx2alpha[self.cur_f[c]]
        return s

    def run(self, runningtime=30):
        start_time = time.time()
        loglikelihood = []
        accepted = []
        while(time.time()-start_time<runningtime):
            f2 = self.generate_f()
            pf2 = self.Pf(f2)
            pf1 = self.Pf(self.cur_f)
            rand = random.random()
            if pf1 == "not exist" and pf2 == "not exist":
                if random.random() < 0.5:
                    accepted.append(True)
                    self.cur_f = f2
                else:
                    accepted.append(False)
            elif pf1 == "not exist":
                accepted.append(True)
                self.cur_f = f2
            elif pf2 == "not exist":
                accepted.append(False)
                pass

            elif rand < min(1, np.exp(pf2 - pf1)):
                self.cur_f = f2
                accepted.append(True)

            else:
                accepted.append(False)
            if self.Pf(self.cur_f) != "not exist":
                loglikelihood.append(self.Pf(self.cur_f))
            else:
                loglikelihood.append(float('nan'))

        return loglikelihood[-1], self.cur_f
        
        
def run(args):
    ciphertext, seed = args
    random.seed(seed)
    mcmc = MCMC(ciphertext=ciphertext)
    loglikelihood, cur_f = mcmc.run(runningtime=10)
    return loglikelihood, cur_f
        
def multi_merge(ciphertext, np=10):
    p = Pool(processes=np)
    data = p.map(run, zip([ciphertext]*np, [time.time()*random.random() for i in range(np)]))
    p.close()
    
    best_loglikelihood = -float('inf')
    best_f = None
    for i in data:
        if(i[0]==i[0] and i[0]>best_loglikelihood):
            best_loglikelihood = i[0]
            best_f = i[1]
    
    final_mcmc = MCMC(ciphertext=ciphertext)
    final_mcmc.cur_f = best_f
    final_mcmc.run(5000)
    return final_mcmc.decode()
    
    



def decode(ciphertext, has_breakpoint):
    if not has_breakpoint:
        plaintext = multi_merge(ciphertext, 10)
    else:
        plaintext = ciphertext
    return plaintext