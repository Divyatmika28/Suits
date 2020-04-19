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