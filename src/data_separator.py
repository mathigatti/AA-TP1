import json
import random as rd

ham = json.load(open('./ham_dev.json'))
spam = json.load(open('./spam_dev.json'))

rd.shuffle(ham)
rd.shuffle(spam)

ham_1 = ham[:int(len(ham)*0.6)]
ham_2 = ham[int(len(ham)*0.6)+1:]

spam_1 = spam[:int(len(spam)*0.6)]
spam_2 = spam[int(len(spam)*0.6)+1:]

with open('training_ham.json', 'w') as outfile:
    json.dump(ham_1, outfile)

with open('training_spam.json', 'w') as outfile:
    json.dump(spam_1, outfile)

with open('testing_ham.json', 'w') as outfile:
    json.dump(ham_2, outfile)

with open('testing_spam.json', 'w') as outfile:
    json.dump(spam_2, outfile)

