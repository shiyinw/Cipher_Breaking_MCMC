import numpy as np
import math, random, time, collections, pickle, itertools
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




def refine(content, words_dict):
    words = content.replace("\n", "").split(" ")
    low_freq_words = {}
    for a in ["k", "j", "z", "q", "x"]:
        low_freq_words[a] = [x for x in words if a in x]

    best_per = dict(zip(["k", "j", "z", "q", "x"], ["k", "j", "z", "q", "x"]))
    best_acc = 0

    for per in list(itertools.permutations(["k", "j", "z", "q", "x"])):
        pi = dict(zip(["k", "j", "z", "q", "x"], per))
        score = 0
        for a in ["k", "j", "z", "q", "x"]:
            score += len([x for x in low_freq_words[a] if x.replace(a, pi[a]) in words_dict[a]])
        if score >= best_acc:
            best_acc = score
            best_per = pi

    content = content.replace("k", "1")
    content = content.replace("j", "2")
    content = content.replace("z", "3")
    content = content.replace("q", "4")
    content = content.replace("x", "5")

    content = content.replace("1", best_per["k"])
    content = content.replace("2", best_per["j"])
    content = content.replace("3", best_per["z"])
    content = content.replace("4", best_per["q"])
    content = content.replace("5", best_per["x"])
    return content


class MCMC:
    def __init__(self, ciphertext):
        self.alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '.']
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
        while(int(time.time()-start_time)<runningtime):
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

            if (len(accepted)>500 and np.sum(accepted[-500:]) == 0):
                break

            if self.Pf(self.cur_f) != "not exist":
                loglikelihood.append(self.Pf(self.cur_f))
            else:
                loglikelihood.append(float('nan'))
           
        return loglikelihood[-1], self.cur_f


class MCMC_B:
    def __init__(self, ciphertext, f1=None, f2=None):
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                    'u', 'v', 'w', 'x', 'y', 'z', ' ', '.']
        self.idx2alpha = dict(zip(range(28), alphabet))
        self.alphabet = alphabet
        self.ciphertext_transition = collections.Counter()
        for a in self.alphabet:
            self.ciphertext_transition[a] = collections.Counter()
        for i in range(1, len(ciphertext), 1):  # including /n
            self.ciphertext_transition[ciphertext[i]][ciphertext[i - 1]] += 1
        self.ciphertext = ciphertext

        self.cur_f1 = f1
        self.cur_f2 = f2

        if (self.cur_f1 == None or self.cur_f2 == None):
            if (self.cur_f1 == None):
                random.shuffle(alphabet)
                self.cur_f1 = dict(zip(alphabet, range(28)))
            if (self.cur_f2 == None):
                random.shuffle(alphabet)
                self.cur_f2 = dict(zip(alphabet, range(28)))
            self.cur_f1, self.cur_f2, self.breakpoint = self.generate_f()

        l = 0
        r = len(self.ciphertext) - 1
        # [minb, maxb]
        # determining the maximum x such that ciphertext[:x] valid
        while (l <= r):
            m = int(l + (r - l) / 2)
            a, b = self.checkvalid(self.ciphertext[m:])
            if (len(a) > 0 and len(b) > 1):
                r = m - 1
            else:
                l = m + 1
        self.minbs = l
        self.leftsets = self.checkvalid(self.ciphertext[:self.minbs])

        l = 0
        r = len(self.ciphertext) - 1
        # [minb, maxb]
        # determining the minimum x such that ciphertext[x:] valid
        while (l <= r):
            m = int(l + (r - l) / 2)
            a, b = self.checkvalid(self.ciphertext[:m])
            if (len(a) > 0 and len(b) > 1):
                l = m + 1
            else:
                r = m - 1
        self.maxbs = r
        self.rightsets = self.checkvalid(self.ciphertext[self.maxbs:])
        self.breakpoint = random.randint(self.minbs, self.maxbs + 1)

        self.ciphertext_transition_left = collections.Counter()
        for a in self.alphabet:
            self.ciphertext_transition_left[a] = collections.Counter()
        for i in range(1, self.minbs, 1):  # including /n
            self.ciphertext_transition_left[ciphertext[i]][ciphertext[i - 1]] += 1

        self.ciphertext_transition_right = collections.Counter()
        for a in self.alphabet:
            self.ciphertext_transition_left[a] = collections.Counter()
        for i in range(self.maxbs+1, len(self.ciphertext), 1):  # including /n
            self.ciphertext_transition_left[ciphertext[i]][ciphertext[i - 1]] += 1

    def Pf(self, code2idx1, code2idx2, breakpoint):
        logPf1 = logP[code2idx1[ciphertext[0]]]

        for a in self.alphabet:
            for b in self.alphabet:
                if self.ciphertext_transition_left[a][b]!=0:
                    if logM[code2idx[a], code2idx[b]] == -float('inf'):
                        return "not exist"
                    else:
                        logPf1 += self.ciphertext_transition_left[a][b] * logM[code2idx[a], code2idx[b]]

        for idx in range(self.minbs, breakpoint, 1):
            if logM[code2idx1[self.ciphertext[idx]], code2idx1[self.ciphertext[idx - 1]]] == -float('inf'):
                logPf1 = "not exist"
                break
            else:
                logPf1 += logM[code2idx1[self.ciphertext[idx]], code2idx1[self.ciphertext[idx - 1]]]

        logPf2 = logP[code2idx2[ciphertext[breakpoint]]]
        for a in self.alphabet:
            for b in self.alphabet:
                if self.ciphertext_transition_right[a][b]!=0:
                    if logM[code2idx[a], code2idx[b]] == -float('inf'):
                        return "not exist"
                    else:
                        logPf2 += self.ciphertext_transition_right[a][b] * logM[code2idx[a], code2idx[b]]
        for idx in range(breakpoint + 1, self.maxbs+1, 1):
            if logM[code2idx2[self.ciphertext[idx]], code2idx2[self.ciphertext[idx - 1]]] == -float('inf'):
                logPf2 = "not exist"
                break
            else:
                logPf2 += logM[code2idx2[self.ciphertext[idx]], code2idx2[self.ciphertext[idx - 1]]]
        return logPf1, logPf2

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

    def decode(self):
        s = ""
        for c in range(self.breakpoint):
            s += self.idx2alpha[self.cur_f1[self.ciphertext[c]]]
        for c in range(self.breakpoint, len(self.ciphertext), 1):
            s += self.idx2alpha[self.cur_f2[self.ciphertext[c]]]
        return s

    def accept(self, newscore, oldscore):
        accept_bool = False
        if newscore == "not exist" and oldscore == "not exist":
            if random.random() < 0.5:
                accept_bool = True
        elif oldscore == "not exist":
            accept_bool = True
        elif newscore == "not exist":
            pass
        elif (newscore - oldscore) > 5:  # speed up for np.exp
            accept_bool = True
        elif random.random() < min(1, np.exp(newscore - oldscore)):
            accept_bool = True
        return accept_bool

    def run(self, runningtime=60):
        start_time = time.time()
        while(time.time()-start_time< runningtime):
            new_f1 = self.generate_f(self.cur_f1)
            new_f2 = self.generate_f(self.cur_f2)
            new_b = int(np.random.normal(loc=self.breakpoint, scale=10))
            pfnew_f1, pfnew_f2 = self.Pf(new_f1, new_f2, new_b)
            pfold_f1, pfold_f2 = self.Pf(self.cur_f1, self.cur_f2, self.breakpoint)

            if self.accept(newscore=pfnew_f1, oldscore=pfold_f1):
                self.cur_f1 = new_f1
            if self.accept(newscore=pfnew_f2, oldscore=pfold_f2):
                self.cur_f2 = new_f2

            if pfnew_f1 == "not exist" or pfnew_f2 == "not exist":
                pfnew_b = "not exist"
            else:
                pfnew_b = pfnew_f1 * pfnew_f2
            if pfold_f1 == "not exist" or pfold_f2 == "not exist":
                pfold_b = "not exist"
            else:
                pfold_b = pfold_f1 * pfold_f2
            if self.accept(newscore=pfnew_b, oldscore=pfold_b):
                self.breakpoint = new_b

        return self.decode()


# class MCMC_B(MCMC):
#     def __init__(self, ciphertext):
#         MCMC.__init__(self, ciphertext)
#         random.shuffle(self.alphabet)
#         self.cur_f1 = dict(zip(self.alphabet, range(28)))
#         random.shuffle(self.alphabet)
#         self.cur_f2 = dict(zip(self.alphabet, range(28)))
#         self.breakpoint = int(len(ciphertext)/2)
#
#     def Pf(self, code2idx, breakpoint):
#         logPf1 = logP[code2idx[ciphertext[breakpoint-1]]]
#         for idx in range(1, breakpoint, 1):
#             if logM[code2idx[idx], code2idx[idx-1]] == -float('inf'):
#                 return "not exist"
#             else:
#                 logPf1 += logM[code2idx[a], code2idx[b]]
#
#         logPf1 = logP[code2idx[ciphertext[breakpoint]]]
#         for idx in range(breakpoint+1, len(self.ciphertext), 1):
#             if logM[code2idx[idx], code2idx[idx-1]] == -float('inf'):
#                 return "not exist"
#             else:
#                 logPf1 += logM[code2idx[a], code2idx[b]]
#         return logPf1*logPf2
#
#     def generate_f(self):
#         a, b = random.sample(self.alphabet, 2)
#         f2 = self.cur_f.copy()
#         f2[a] = self.cur_f[b]
#         f2[b] = self.cur_f[a]
#         return f2
#
#     def decode(self):
#         s = ""
#         for c in self.ciphertext:
#             s += self.idx2alpha[self.cur_f[c]]
#         return s
#
#     def run(self, runningtime=30):
#         start_time = time.time()
#         loglikelihood = []
#         accepted = []
#         while(time.time()-start_time<runningtime):
#             f2 = self.generate_f()
#             pf2 = self.Pf(f2)
#             pf1 = self.Pf(self.cur_f)
#             rand = random.random()
#             if pf1 == "not exist" and pf2 == "not exist":
#                 if random.random() < 0.5:
#                     accepted.append(True)
#                     self.cur_f = f2
#                 else:
#                     accepted.append(False)
#             elif pf1 == "not exist":
#                 accepted.append(True)
#                 self.cur_f = f2
#             elif pf2 == "not exist":
#                 accepted.append(False)
#                 pass
#
#             elif rand < min(1, np.exp(pf2 - pf1)):
#                 self.cur_f = f2
#                 accepted.append(True)
#
#             else:
#                 accepted.append(False)
#             if self.Pf(self.cur_f) != "not exist":
#                 loglikelihood.append(self.Pf(self.cur_f))
#             else:
#                 loglikelihood.append(float('nan'))
#
#         return loglikelihood[-1], self.cur_f
        
        
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
    final_mcmc.run(runningtime=30)
    return final_mcmc.decode(), final_mcmc.cur_f

def checkvalid(content):
    transition = collections.Counter()
    for i in Alphabet:
        transition[i] = collections.Counter()
    for i in range(1, len(content), 1):  # including /n
        transition[content[i - 1]][content[i]] += 1

    notrepeat = []  # ' ' & '.'
    for a in Alphabet:
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

def breakpoint_range(content):
    l = 0
    r = len(content) - 1
    # [minb, maxb]
    # determining the maximum x such that ciphertext[:x] valid
    while (l <= r):
        m = int(l + (r - l) / 2)
        a, b = checkvalid(content[m:])
        if (len(a) > 0 and len(b) > 1):
            r = m - 1
        else:
            l = m + 1
    minbs = l
    leftsets = checkvalid(content[:minbs])

    l = 0
    r = len(content) - 1
    # [minb, maxb]
    # determining the minimum x such that ciphertext[x:] valid
    while (l <= r):
        m = int(l + (r - l) / 2)
        a, b = checkvalid(content[:m])
        if (len(a) > 0 and len(b) > 1):
            l = m + 1
        else:
            r = m - 1
    maxbs = r
    rightsets = checkvalid(content[maxbs:])

    return minbs, maxbs, leftsets, rightsets

def decode(ciphertext, has_breakpoint):
    if not has_breakpoint:
        plaintext, _ = multi_merge(ciphertext, 10)
    else:
        minb, maxb, leftsets, rightsets = breakpoint_range(ciphertext)
        lefttext, single_f1 = multi_merge(ciphertext[:minb], 10)
        righttext, single_f2 = multi_merge(ciphertext[maxb:], 10)
        try:
            mcmc = MCMC_B(ciphertext=ciphertext, f1=single_f1, f2=single_f2)
            plaintext = mcmc.run(runningtime=60)
        except:
            plaintext = lefttext + ciphertext[minb:maxb] + righttext

    try:
        with open("words.pickle", "rb") as f:
            words = pickle.load(f)
        plaintext = refine(plaintext, words)
    except Exception as e:
        print(e)
        pass
    return plaintext