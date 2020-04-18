import os, json
from nltk.corpus import stopwords
from collections import defaultdict

categorical_phrases = dict()

categorical_phrases['Introduction'] = [
    'no', 'appelant', 'date', 'etat', 'depose', 'gouvernement', 'enregistrement', 'base', 'declare', 'avis', 'periode', 'region',
    'janvier', 'fevrier', 'mars', 'avril', 'mai', 'juin', 'jullet', 'aout', 'septembre', 'octobre', 'novembre', 'decembre',
    'demandeur', 'base', 'cour', 'depose', 'delivre', 'constitution',
    'societe', 'vertu', 'article', 'justifier', 'annee', 'appel',
    'application', 'revision', 'decision', 'requete', 'dossier', 'depose', 'reclamation', 'declaration', 'ordonnance', 'judgement'
]

categorical_phrases['Context'] = [
    'service', 'montant', 'valeur', 'section', 'marchandises', 'credit', 'licence', 'fourni', 'destinataire', 'pays',
    'quota', 'conteste',
    'contention', 'section', 'termes', 'avocat', 'devant', 'charge', 'facture', 'sans', 'facture', 'fourni',
    'truibunal', 'defendeur', 'defendeurs', 'juge', 'juges', 'magistrat', 'magistrats', 'chambre', 'proces', 'tribunal',
    'conseiller', 'indiquer', 'preoccupation', 'demande', 'demandes', 'emplyeur'
]

categorical_phrases['Analysis'] = [
    'dispositions', 'disposition', 'loi', 'section', 'chapitre', 'explication', 'comission', 'agent', 'successions',
    'service', 'reserve', 'repute', 'successions', 'but', 'buts', 'cas', 'conclusion', 'observe', 'ce tribunal',
    'question', 'pose', 'question', 'fait', 'ete', 'code', 'attendu', 'arret', 'motifs',
    'civile', 'commerciale', 'criminelle', 'sociale', 'juri', 'attaque',
    'examen', 'section', 'no', 'acte', 'conformement', 'etat', 'etats', 'declare', 'present', 'actuel'
]
# add years
years = [str(x) for x in list(range(1700, 2100))]
categorical_phrases['Analysis'].extend(years)

categorical_phrases['Conclusion'] = [
    'decision', 'rejeter', 'rejete', 'rejetant', 'soutenu', 'rejete', 'autorise', 'adopte', 'affaire', 'considere', 'part',
    'jugement', 'consequence', 'modifie', 'permettre', 'decret', 'section', 'accueilli', 'partie', 'cour'
]

if __name__ == '__main__':
    print('starting main...')
    i = 0
    words_freq = defaultdict(int)
    stop_words = list(set(stopwords.words('french')))
    punc = ['.', ':', ',', ';', '-', 'a', 'l\'', 'd\'', 'qu\'', 'n\'']
    stop_words += punc

    for path, _, files in os.walk('../CASS-dataset/cleaned_files'):
        for f in files:
            i += 1
            if i > 10000:
                break
            with open(os.path.join(path, f), 'r', encoding='utf-8') as r_file:
                full_text = r_file.read().split('@summary ')
                text = full_text[0]
                summ = full_text[1]
                word_text = text.split()

                for word in word_text:
                    if word not in stop_words:
                        words_freq[word] += 1

    print(i)
    words_freq = sorted(words_freq.items(), reverse=True, key=lambda x: x[1])

    with open('./category_words.txt', 'w', encoding='utf-8') as cw:
        for i in range(1000):
            word = words_freq[i][0]
            freq = words_freq[i][1]
            wrote = False
            for category, word_match in categorical_phrases.items():
                if word in word_match:
                    cw.write('\nCategory: '+ category)
                    cw.write('\nWord, frequqnecy: '+ word+ ' '+ str(freq))
                    # print('Category: ', category)
                    # print('Word, frequqnecy: ', word, freq)
                    wrote = True
            if wrote == False:
                cw.write('\nNot in category: ' + word)


    with open('./key_words.txt', 'w', encoding='utf-8') as f:
        json.dump(words_freq, f)