import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

class TextFeatureExtractor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def extract_text_stats(self, texts):
        features = []
        for text in texts:
            stats = {
                'char_count': len(text),
                'word_count': len(text.split()),
                'sentence_count': len(re.split(r'[.!?]+', text)),
                'avg_word_length': np.mean([len(word) for word in text.split()]),
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text),
                'punctuation_count': sum(1 for c in text if c in '.,!?;:')
            }
            features.append(stats)
        return pd.DataFrame(features)
    
    def get_tfidf_features(self, texts):
        return self.tfidf.fit_transform(texts)
    
    def extract_ngrams(self, text, n=2):
        words = text.lower().split()
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        return Counter(ngrams)