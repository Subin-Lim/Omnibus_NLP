!pip install transformers
!pip install konlpy

from transformers import BertTokenizer, BertForSequenceClassification
import torch
from konlpy.tag import Komoran

class IntentDetector:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained('monologg/kobert')
        self.tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
        self.intent_map = {
            0: '버스 번호 검색',
            1: '노선 없음'
        }
        self.komoran = Komoran()

    def recognize_intent_and_destination(self, text):
        # 텍스트 전처리
        input_text = self.preprocess_text(text)

        # BERT 토크나이저를 사용해 텍스트를 토큰화하고 ID로 변환
        input_ids = torch.tensor([self.tokenizer.encode(input_text, add_special_tokens=True)])

        # BERT 모델로 의도 예측
        with torch.no_grad():
            output = self.model(input_ids)[0]
        intent_id = torch.argmax(output, dim=1).item()

        # Komoran으로 형태소 분석을 통해 목적지 추출
        tokens = self.komoran.pos(input_text)
        print("토큰화 및 품사 태깅 결과:", tokens)

        # 목적지 추출
        destination = self.extract_destination(tokens)

        # 목적지가 추출된 경우 의도를 무조건 '버스 번호 검색'으로 설정
        if destination:
            intent = '버스 번호 검색'
        else:
            intent = self.intent_map[intent_id]

        return intent, destination

    def preprocess_text(self, text):
        # 텍스트를 목적지 추출을 위해 전처리
        text = text.replace('가려고', ' ')
        text = text.replace('가고 싶어', ' ')
        text = text.replace('가요', ' ')
        text = text.replace('갈래', ' ')
        text = text.replace('가야 돼', ' ')
        text = text.replace('가려면', ' ')
        return text

    def extract_destination(self, tokens):
        # 고유명사(NNP)나 지명 관련 일반명사(NNG)를 목적지로 추출
        for token, pos in tokens:
            if pos in ['NNP'] and '역' in token:  # "역"이 포함된 경우를 목적지로 간주
                return token
            if pos == 'NNP':  # 고유명사를 목적지로 간주
                return token
        return None

# 테스트 코드
intent_detector = IntentDetector()
texts = [
    "저녁에 용산 가려고 하는데 몇 번 타야돼?",
    "강남역 가는 노선 알려줘",
    "경복궁 가고 싶어",
    "서울역 갈래",
    "배고프다 배고프고 졸리고 집에 가고 싶어",
    "구로디지털단지역 가려면 몇번 버스 타야해?"
]

for text in texts:
    intent, destination = intent_detector.recognize_intent_and_destination(text)
    print(f"'{text}' - 의도: {intent}, 목적지: {destination}")
