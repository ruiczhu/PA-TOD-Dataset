import pandas as pd
import re
import nltk
from collections import Counter
from textstat import flesch_reading_ease, flesch_kincaid_grade
import numpy as np
# 添加新的导入
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm  # 添加进度条
import time

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    print("SpaCy模型未安装，某些功能可能受限")
    nlp = None

# 下载必要的NLTK数据
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('opinion_lexicon', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import opinion_lexicon, stopwords, wordnet
    from nltk.corpus.reader.wordnet import WordNetError
except:
    print("NLTK资源下载失败，将使用基础词汇列表")

data = pd.read_csv('personality_behavior_val/processed_data_encoded.csv')

def filter_by_personality(trait=None, level=None, **additional_conditions):
    """
    根据性格特征和程度筛选数据
    
    参数:
        trait: str, 性格特征 ('openness', 'agreeableness', 'conscientiousness', 'extraversion', 'neuroticism')
        level: str, 程度 ('high', 'low')
        **additional_conditions: 其他额外的过滤条件
    
    返回:
        pandas.DataFrame: 符合条件的数据
    
    示例:
        # 获取openness特征的所有数据
        result = filter_by_personality(trait='openness')
        
        # 获取openness特征且为low程度的数据
        result = filter_by_personality(trait='openness', level='low')
        
        # 获取high程度的所有数据
        result = filter_by_personality(level='high')
    """
    filtered_data = data.copy()
    
    # 根据性格特征筛选
    if trait:
        trait_column = f'trait_{trait}'
        if trait_column in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[trait_column] == True]
        else:
            print(f"警告: 性格特征列 '{trait_column}' 不存在于数据集中")
            return pd.DataFrame()
    
    # 根据程度筛选
    if level:
        level_column = f'level_{level}'
        if level_column in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[level_column] == True]
        else:
            print(f"警告: 程度列 '{level_column}' 不存在于数据集中")
            return pd.DataFrame()
    
    # 应用其他额外条件
    for column, value in additional_conditions.items():
        if column in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[column] == value]
        else:
            print(f"警告: 列 '{column}' 不存在于数据集中")
    
    return filtered_data

class PersonalityBehaviorEvaluator:
    """
    对筛选后的数据进行行为特征评估
    """
    
    def __init__(self, data):
        self.data = data
        self.setup_word_lists()
        # 初始化情感分析器
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def get_wordnet_synonyms(self, word, pos_tag=None):
        """使用WordNet获取同义词"""
        synonyms = set()
        try:
            synsets = wordnet.synsets(word, pos=pos_tag)
            for synset in synsets[:3]:  # 限制前3个意思避免过度扩展
                for lemma in synset.lemmas():
                    synonym = lemma.name().lower().replace('_', ' ')
                    if len(synonym.split()) == 1:  # 只要单词，不要短语
                        synonyms.add(synonym)
        except (WordNetError, AttributeError):
            pass
        return synonyms
    
    def expand_word_list(self, base_words, max_expansions=200):
        """使用WordNet扩展词汇列表"""
        expanded = set(base_words)
        for word in base_words:
            synonyms = self.get_wordnet_synonyms(word)
            expanded.update(synonyms)
            if len(expanded) > max_expansions:
                break
        return expanded
    
    def setup_word_lists(self):
        """设置各类词汇列表 - 使用词库和自动扩展"""
        
        # 基础种子词 - 更精准的核心词汇
        achievement_seeds = {
            'accomplish', 'achieve', 'success', 'goal', 'complete', 'finish', 'win', 'excel',
            'master', 'triumph', 'victory', 'progress', 'improve', 'develop', 'build',
            'create', 'solve', 'manage', 'lead', 'organize', 'strategy', 'target'
        }
        
        social_seeds = {
            'friend', 'social', 'together', 'share', 'meet', 'group', 'community',
            'relationship', 'connect', 'interact', 'communicate', 'collaborate',
            'team', 'family', 'love', 'trust', 'support', 'help', 'care'
        }
        
        aggressive_seeds = {
            'angry', 'hate', 'fight', 'attack', 'destroy', 'violence', 'aggressive',
            'hostile', 'rage', 'fury', 'battle', 'war', 'conflict', 'argue',
            'threat', 'harm', 'cruel', 'brutal', 'fierce'
        }
        
        politeness_seeds = {
            'please', 'thank', 'sorry', 'excuse', 'pardon', 'welcome', 'appreciate',
            'respect', 'courtesy', 'kindly', 'gently', 'humble', 'modest',
            'considerate', 'thoughtful', 'caring', 'gentle', 'polite'
        }
        
        uncertainty_seeds = {
            'maybe', 'perhaps', 'possibly', 'might', 'could', 'uncertain', 'unsure',
            'doubt', 'unclear', 'confused', 'probably', 'likely', 'suppose',
            'guess', 'assume', 'wonder', 'question', 'hesitant', 'tentative'
        }
        
        # 使用WordNet扩展词汇列表，并确保每个类别都有足够的词汇
        try:
            self.achievement_words = self.expand_word_list(achievement_seeds, 150)
            self.social_words = self.expand_word_list(social_seeds, 200)
            self.aggressive_words = self.expand_word_list(aggressive_seeds, 150)
            self.politeness_words = self.expand_word_list(politeness_seeds, 150)
            self.uncertainty_words = self.expand_word_list(uncertainty_seeds, 150)
            
            # 打印词汇表大小以便调试
            print(f"词汇表大小: "
                  f"成就={len(self.achievement_words)}, "
                  f"社交={len(self.social_words)}, "
                  f"攻击性={len(self.aggressive_words)}, "
                  f"礼貌={len(self.politeness_words)}")
                  
        except Exception as e:
            print(f"扩展词汇列表时出错: {e}")
            # 使用基础种子词作为后备
            self.achievement_words = achievement_seeds
            self.social_words = social_seeds
            self.aggressive_words = aggressive_seeds
            self.politeness_words = politeness_seeds
            self.uncertainty_words = uncertainty_seeds
        
        # 确认性词语 - 针对寻求他人确认的表达
        confirmation_base = {
            'confirm', 'verify', 'check', 'clarify', 'validate', 'ensure',
            'double-check', 'make sure', 'understand correctly', 'mean',
            'saying', 'suggesting', 'implying', 'indicate', 'refer',
            'right', 'correct', 'accurate', 'true', 'proper'
        }
        # 添加确认性短语的关键词
        confirmation_phrases = {
            'so', 'therefore', 'thus', 'hence', 'meaning', 'means',
            'saying', 'tell', 'suggest', 'imply', 'indicate'
        }
        confirmation_base.update(confirmation_phrases)
        self.confirmation_words = self.expand_word_list(confirmation_base, 100)
    
    def calculate_ttr(self, text):
        """计算词汇多样性（Type-Token Ratio）"""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) == 0:
            return 0
        unique_words = len(set(words))
        total_words = len(words)
        return unique_words / total_words
    
    def count_achievement_words(self, text):
        """计算与成就相关的词语"""
        words = re.findall(r'\b\w+\b', text.lower())
        return sum(1 for word in words if word in self.achievement_words)
    
    def calculate_avg_utterance_length(self, text):
        """计算平均话语长度"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0
        
        word_counts = []
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence)
            word_counts.append(len(words))
        
        return np.mean(word_counts) if word_counts else 0
    
    def count_questions(self, text):
        """计算提问行为"""
        return text.count('?')
    
    def count_confirmation_words(self, text):
        """计算确认性词语"""
        words = re.findall(r'\b\w+\b', text.lower())
        return sum(1 for word in words if word in self.confirmation_words)
    
    def count_social_words(self, text):
        """计算社交性词语"""
        words = re.findall(r'\b\w+\b', text.lower())
        return sum(1 for word in words if word in self.social_words)
    
    def count_aggressive_words(self, text):
        """计算攻击性词汇"""
        words = re.findall(r'\b\w+\b', text.lower())
        return sum(1 for word in words if word in self.aggressive_words)
    
    def count_politeness_words(self, text):
        """计算礼貌用语"""
        words = re.findall(r'\b\w+\b', text.lower())
        return sum(1 for word in words if word in self.politeness_words)
    
    def count_uncertainty_words(self, text):
        """计算不确定性词汇"""
        words = re.findall(r'\b\w+\b', text.lower())
        return sum(1 for word in words if word in self.uncertainty_words)
    
    def calculate_mtld(self, text):
        """计算Moving-Average Type-Token Ratio (MTLD)"""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 50:  # MTLD需要足够的文本长度
            return self.calculate_ttr(text)
        
        # 简化的MTLD实现
        ttr_threshold = 0.72
        word_count = 0
        type_count = 0
        factor_count = 0
        word_types = set()
        
        for word in words:
            word_count += 1
            word_types.add(word)
            type_count = len(word_types)
            
            if type_count / word_count <= ttr_threshold:
                factor_count += 1
                word_types.clear()
                word_count = 0
                type_count = 0
        
        return len(words) / factor_count if factor_count > 0 else len(words)
    
    def calculate_lexical_density(self, text):
        """计算词汇密度（内容词与总词数的比例）"""
        if not nlp:
            return 0
        
        doc = nlp(text)
        content_words = 0
        total_words = 0
        
        for token in doc:
            if token.is_alpha:
                total_words += 1
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                    content_words += 1
        
        return content_words / total_words if total_words > 0 else 0
    
    def analyze_vader_sentiment(self, text):
        """使用VADER进行情感分析"""
        scores = self.vader_analyzer.polarity_scores(text)
        return scores

    
    def calculate_lexical_richness(self, text):
        """增强版词汇丰富度计算，包含多种指标"""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) == 0:
            return {
                'vocabulary_size': 0,
                'word_frequency_variance': 0
            }
        
        # 词汇量大小
        unique_words = set(words)
        vocabulary_size = len(unique_words)
        
        # 词频方差（衡量词汇使用的均匀程度）
        word_counts = Counter(words)
        frequencies = list(word_counts.values())
        word_frequency_variance = np.var(frequencies) if len(frequencies) > 1 else 0
        
        return {
            'vocabulary_size': vocabulary_size,
            'word_frequency_variance': word_frequency_variance
        }

    def evaluate_all(self, text_column='text'):
        """
        对所有数据进行完整评估
        
        参数:
            text_column: str, 包含文本内容的列名
        
        返回:
            pandas.DataFrame: 包含所有评估指标的结果
        """
        if text_column not in self.data.columns:
            print(f"警告: 文本列 '{text_column}' 不存在于数据集中")
            return pd.DataFrame()
        
        results = []
        total_rows = len(self.data)
        
        print(f"开始评估 {total_rows} 条记录，共 {len(self.get_all_metrics())} 个指标...")
        
        # 使用tqdm创建进度条
        with tqdm(total=total_rows, desc="评估进度", unit="条记录") as pbar:
            for idx, row in self.data.iterrows():
                start_time = time.time()
                text = str(row[text_column])
                
                # 基础指标
                result = {
                    'index': idx,
                    'ttr': self.calculate_ttr(text),
                    'mtld': self.calculate_mtld(text),
                    'lexical_density': self.calculate_lexical_density(text),
                    'achievement_words_count': self.count_achievement_words(text),
                    'avg_utterance_length': self.calculate_avg_utterance_length(text),
                    'questions_count': self.count_questions(text),
                    'confirmation_words_count': self.count_confirmation_words(text),
                    'social_words_count': self.count_social_words(text),
                    'aggressive_words_count': self.count_aggressive_words(text),
                    'politeness_words_count': self.count_politeness_words(text),
                    'uncertainty_words_count': self.count_uncertainty_words(text)
                }
                
                # 增强版词汇丰富度指标
                lexical_richness = self.calculate_lexical_richness(text)
                result.update(lexical_richness)
                
                # 情感分析
                vader_scores = self.analyze_vader_sentiment(text)
                result.update({
                    'vader_compound': vader_scores['compound'],
                    'vader_positive': vader_scores['pos'],
                    'vader_negative': vader_scores['neg']
                })
                
                results.append(result)
                
                # 更新进度条
                processing_time = time.time() - start_time
                pbar.set_postfix({
                    '当前记录': f'{idx}',
                    '处理时间': f'{processing_time:.2f}s',
                    '剩余估算': f'{(total_rows - len(results)) * processing_time:.1f}s'
                })
                pbar.update(1)
        
        print(f"评估完成！处理了 {len(results)} 条记录")
        return pd.DataFrame(results)
    
    def get_all_metrics(self):
        """获取所有评估指标的列表"""
        return [
            'ttr', 'mtld', 'lexical_density', 'vocabulary_size', 'word_frequency_variance', 
            'achievement_words_count', 'avg_utterance_length', 
            'questions_count', 'confirmation_words_count', 'social_words_count', 
            'aggressive_words_count', 'politeness_words_count', 'uncertainty_words_count', 
            'vader_compound', 'vader_positive', 'vader_negative'
        ]