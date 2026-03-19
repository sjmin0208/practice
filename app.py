import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="증상 기반 질병·치료법 대시보드",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 면책 조항 ─────────────────────────────────────────────────────────────────
st.warning(
    "⚠️ **면책 조항**: 이 도구는 의료기기가 아니며 진단·처방을 대체하지 않습니다. "
    "약품 복용 전 반드시 의사·약사와 상담하세요. 참고용 건강 정보 도구입니다."
)

# ════════════════════════════════════════════════════════════════════════════
#  데이터 정의
# ════════════════════════════════════════════════════════════════════════════

DISEASE_KR = {
    "Fungal infection": "곰팡이 감염", "Allergy": "알레르기", "GERD": "위식도역류",
    "Chronic cholestasis": "만성 담즙정체", "Drug Reaction": "약물 반응",
    "Peptic ulcer disease": "소화성 궤양", "AIDS": "에이즈", "Diabetes": "당뇨",
    "Gastroenteritis": "위장염", "Bronchial Asthma": "기관지 천식",
    "Hypertension": "고혈압", "Migraine": "편두통",
    "Cervical spondylosis": "경추 척추증",
    "Paralysis (brain hemorrhage)": "뇌출혈/마비", "Jaundice": "황달",
    "Malaria": "말라리아", "Chicken pox": "수두", "Dengue": "뎅기열",
    "Typhoid": "장티푸스", "hepatitis A": "A형 간염", "Hepatitis B": "B형 간염",
    "Hepatitis C": "C형 간염", "Hepatitis D": "D형 간염", "Hepatitis E": "E형 간염",
    "Alcoholic hepatitis": "알코올성 간염", "Tuberculosis": "결핵",
    "Common Cold": "감기", "Pneumonia": "폐렴",
    "Dimorphic hemorrhoids(piles)": "치질", "Heart attack": "심근경색",
    "Varicose veins": "정맥류", "Hypothyroidism": "갑상선 기능 저하",
    "Hyperthyroidism": "갑상선 기능 항진", "Hypoglycemia": "저혈당",
    "Osteoarthritis": "골관절염", "Arthritis": "관절염",
    "(vertigo) Paroxysmal Positional Vertigo": "이석증(어지럼증)",
    "Acne": "여드름", "Urinary tract infection": "요로 감염",
    "Psoriasis": "건선", "Impetigo": "농가진",
}

SYMPTOM_KR = {
    "itching": "가려움증", "skin_rash": "피부 발진",
    "nodal_skin_eruptions": "결절성 피부 발진",
    "continuous_sneezing": "지속적 재채기", "shivering": "떨림", "chills": "오한",
    "joint_pain": "관절통", "stomach_pain": "복통", "acidity": "위산 과다",
    "ulcers_on_tongue": "구내염", "muscle_wasting": "근육 소모", "vomiting": "구토",
    "burning_micturition": "배뇨 시 화끈감", "spotting_urination": "혈뇨",
    "fatigue": "피로감", "weight_gain": "체중 증가", "anxiety": "불안감",
    "cold_hands_and_feets": "손발 냉증", "mood_swings": "감정 기복",
    "weight_loss": "체중 감소", "restlessness": "안절부절", "lethargy": "무기력증",
    "patches_in_throat": "인후 반점", "irregular_sugar_level": "혈당 불규칙",
    "cough": "기침", "high_fever": "고열", "sunken_eyes": "움푹 꺼진 눈",
    "breathlessness": "호흡 곤란", "sweating": "발한(땀)", "dehydration": "탈수",
    "indigestion": "소화불량", "headache": "두통", "yellowish_skin": "황달(피부)",
    "dark_urine": "짙은 소변", "nausea": "메스꺼움", "loss_of_appetite": "식욕 부진",
    "pain_behind_the_eyes": "눈 뒤 통증", "back_pain": "허리 통증",
    "constipation": "변비", "abdominal_pain": "복부 통증", "diarrhoea": "설사",
    "mild_fever": "미열", "yellow_urine": "황색 소변", "yellowing_of_eyes": "눈 황달",
    "acute_liver_failure": "급성 간부전", "fluid_overload": "체액 과다",
    "swelling_of_stomach": "복부 팽만", "swelled_lymph_nodes": "림프절 부종",
    "malaise": "전신 불쾌감", "blurred_and_distorted_vision": "시야 흐림",
    "phlegm": "가래", "throat_irritation": "인후 자극",
    "redness_of_eyes": "눈 충혈", "sinus_pressure": "부비동 압박",
    "runny_nose": "콧물", "congestion": "코막힘", "chest_pain": "흉통",
    "weakness_in_limbs": "사지 무력감", "fast_heart_rate": "빠른 심박수",
    "pain_during_bowel_movements": "배변 통증",
}

# 증상-질병 매핑 (Kaggle itachi9604/kaushil268 기반)
DISEASE_SYMPTOMS = {
    "Fungal infection":      ["itching","skin_rash","nodal_skin_eruptions","fatigue"],
    "Allergy":               ["continuous_sneezing","chills","fatigue","cough","redness_of_eyes","sinus_pressure","runny_nose","congestion","headache"],
    "GERD":                  ["stomach_pain","acidity","vomiting","cough","chest_pain","indigestion","headache","nausea"],
    "Chronic cholestasis":   ["itching","vomiting","fatigue","weight_loss","abdominal_pain","yellowish_skin","dark_urine","nausea"],
    "Drug Reaction":         ["itching","skin_rash","stomach_pain","vomiting","burning_micturition"],
    "Peptic ulcer disease":  ["vomiting","indigestion","loss_of_appetite","abdominal_pain","nausea"],
    "AIDS":                  ["muscle_wasting","fatigue","weight_loss","patches_in_throat","sweating","malaise","swelled_lymph_nodes"],
    "Diabetes":              ["fatigue","weight_loss","restlessness","lethargy","irregular_sugar_level","blurred_and_distorted_vision","weight_gain"],
    "Gastroenteritis":       ["vomiting","sunken_eyes","dehydration","diarrhoea","nausea"],
    "Bronchial Asthma":      ["fatigue","cough","breathlessness","phlegm","chest_pain"],
    "Hypertension":          ["headache","chest_pain","fatigue"],
    "Migraine":              ["headache","nausea","vomiting","blurred_and_distorted_vision","pain_behind_the_eyes","mood_swings"],
    "Cervical spondylosis":  ["back_pain","weakness_in_limbs"],
    "Paralysis (brain hemorrhage)": ["vomiting","headache","weakness_in_limbs","chest_pain","breathlessness"],
    "Jaundice":              ["itching","vomiting","fatigue","weight_loss","high_fever","yellowish_skin","dark_urine","abdominal_pain","yellowing_of_eyes"],
    "Malaria":               ["chills","vomiting","high_fever","sweating","headache","nausea","diarrhoea"],
    "Chicken pox":           ["itching","skin_rash","fatigue","lethargy","high_fever","headache","loss_of_appetite","mild_fever","swelled_lymph_nodes","malaise"],
    "Dengue":                ["skin_rash","chills","joint_pain","vomiting","fatigue","high_fever","headache","nausea","loss_of_appetite","pain_behind_the_eyes","back_pain","malaise"],
    "Typhoid":               ["chills","vomiting","fatigue","high_fever","headache","nausea","constipation","abdominal_pain","diarrhoea"],
    "hepatitis A":           ["joint_pain","vomiting","yellowish_skin","dark_urine","nausea","loss_of_appetite","abdominal_pain","diarrhoea","mild_fever","yellowing_of_eyes"],
    "Hepatitis B":           ["itching","fatigue","lethargy","yellowish_skin","dark_urine","loss_of_appetite","abdominal_pain","malaise","yellowing_of_eyes"],
    "Hepatitis C":           ["fatigue","yellowish_skin","nausea","loss_of_appetite"],
    "Hepatitis D":           ["joint_pain","vomiting","fatigue","yellowish_skin","dark_urine","nausea","loss_of_appetite","abdominal_pain","yellowing_of_eyes"],
    "Hepatitis E":           ["joint_pain","vomiting","fatigue","high_fever","yellowish_skin","dark_urine","nausea","loss_of_appetite","abdominal_pain","yellowing_of_eyes","acute_liver_failure"],
    "Alcoholic hepatitis":   ["vomiting","yellowish_skin","abdominal_pain","swelling_of_stomach","fluid_overload"],
    "Tuberculosis":          ["chills","vomiting","fatigue","weight_loss","cough","high_fever","breathlessness","sweating","loss_of_appetite","mild_fever","swelled_lymph_nodes","malaise","phlegm"],
    "Common Cold":           ["continuous_sneezing","chills","fatigue","cough","headache","runny_nose","congestion","mild_fever","malaise","throat_irritation"],
    "Pneumonia":             ["chills","fatigue","cough","high_fever","breathlessness","sweating","malaise","phlegm","chest_pain","fast_heart_rate"],
    "Dimorphic hemorrhoids(piles)": ["constipation","pain_during_bowel_movements"],
    "Heart attack":          ["vomiting","breathlessness","sweating","chest_pain","fast_heart_rate"],
    "Varicose veins":        ["fatigue"],
    "Hypothyroidism":        ["fatigue","weight_gain","cold_hands_and_feets","mood_swings","lethargy"],
    "Hyperthyroidism":       ["fatigue","mood_swings","weight_loss","restlessness","sweating","fast_heart_rate"],
    "Hypoglycemia":          ["fatigue","weight_loss","restlessness","cold_hands_and_feets","sweating","irregular_sugar_level","anxiety","blurred_and_distorted_vision","fast_heart_rate"],
    "Osteoarthritis":        ["joint_pain","back_pain"],
    "Arthritis":             ["joint_pain","swelled_lymph_nodes"],
    "(vertigo) Paroxysmal Positional Vertigo": ["vomiting","headache","nausea"],
    "Acne":                  ["skin_rash"],
    "Urinary tract infection": ["burning_micturition","spotting_urination"],
    "Psoriasis":             ["skin_rash","joint_pain"],
    "Impetigo":              ["skin_rash","high_fever"],
}

# ── 치료 정보 데이터베이스 ────────────────────────────────────────────────────
# 구조: { 질병명: { drugs, treatments, folk_remedies, warning, urgency } }
# urgency: "즉시 병원" | "빠른 진료" | "경과 관찰"

TREATMENT_DB = {
    "Fungal infection": {
        "drugs": [
            {"name": "클로트리마졸 (Clotrimazole)", "type": "항진균제", "note": "피부 국소 적용, 약국 구매 가능"},
            {"name": "테르비나핀 (Terbinafine)", "type": "항진균제", "note": "발무좀·손발톱 진균증에 효과적"},
            {"name": "플루코나졸 (Fluconazole)", "type": "항진균제 (경구)", "note": "전신 감염 시 의사 처방 필요"},
        ],
        "treatments": ["감염 부위 청결·건조 유지", "통기성 좋은 면 소재 착용", "수건·양말 공유 금지", "심한 경우 피부과 전문의 상담"],
        "folk_remedies": ["티트리 오일 희석 후 국소 도포 (항진균 효과 연구됨)", "애플사이다 식초 희석액으로 환부 세척", "마늘즙 도포 (알리신 성분의 항진균 작용)"],
        "urgency": "경과 관찰",
        "urgency_color": "green",
    },
    "Allergy": {
        "drugs": [
            {"name": "세티리진 (Cetirizine)", "type": "항히스타민제", "note": "지르텍 등, 약국 구매 가능, 졸림 적음"},
            {"name": "로라타딘 (Loratadine)", "type": "항히스타민제", "note": "클라리틴 등, 비졸림성, 일 1회"},
            {"name": "펙소페나딘 (Fexofenadine)", "type": "항히스타민제", "note": "알레그라 등, 운전 시에도 사용 가능"},
            {"name": "나살 스테로이드 스프레이", "type": "국소 스테로이드", "note": "코 증상에 효과적, 처방 필요"},
        ],
        "treatments": ["알레르겐(원인 물질) 회피", "공기청정기 사용", "외출 후 세안·샤워", "꽃가루 많은 날 마스크 착용"],
        "folk_remedies": ["꿀 (지역산) 소량 섭취 — 꽃가루 면역 형성 가설", "생강차 섭취 (항염 효과)", "쿼세틴 함유 식품 섭취 (양파, 사과)"],
        "urgency": "경과 관찰",
        "urgency_color": "green",
    },
    "GERD": {
        "drugs": [
            {"name": "오메프라졸 (Omeprazole)", "type": "PPI (양성자 펌프 억제제)", "note": "로섹 등, 위산 분비 강력 억제"},
            {"name": "란소프라졸 (Lansoprazole)", "type": "PPI", "note": "프리로섹 등, 공복 복용 권장"},
            {"name": "파모티딘 (Famotidine)", "type": "H2 차단제", "note": "약국 구매 가능, 즉각 효과"},
            {"name": "탄산칼슘 제산제", "type": "제산제", "note": "겔포스 등, 즉각 증상 완화용"},
        ],
        "treatments": ["취침 2~3시간 전 식사 금지", "침대 머리쪽 15cm 높이기", "소식·천천히 씹기", "카페인·알코올·고지방 음식 제한", "적정 체중 유지"],
        "folk_remedies": ["알로에베라 주스 (위 점막 보호)", "생강차 소량 섭취", "베이킹소다 물 1/2 티스푼 (응급 제산)"],
        "urgency": "경과 관찰",
        "urgency_color": "green",
    },
    "Peptic ulcer disease": {
        "drugs": [
            {"name": "오메프라졸 (Omeprazole)", "type": "PPI", "note": "위산 억제로 궤양 치유 촉진"},
            {"name": "수크랄페이트 (Sucralfate)", "type": "위점막 보호제", "note": "궤양 부위 보호막 형성"},
            {"name": "아목시실린+클라리스로마이신", "type": "항생제 병합요법", "note": "H.pylori 균 제거, 반드시 의사 처방"},
        ],
        "treatments": ["NSAIDs(이부프로펜 등) 복용 중단", "금주·금연", "스트레스 관리", "자극적 음식 회피", "H.pylori 검사 권장"],
        "folk_remedies": ["양배추즙 (비타민U 성분, 위 점막 보호 연구됨)", "꿀 (항균·항산화)", "감초 DGL 제품 (위 점막 보호)"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "Diabetes": {
        "drugs": [
            {"name": "메트포르민 (Metformin)", "type": "혈당강하제 1차선택", "note": "인슐린 저항성 개선, 처방 필요"},
            {"name": "글리피짓 (Glipizide)", "type": "설포닐우레아계", "note": "인슐린 분비 촉진, 처방 필요"},
            {"name": "인슐린 (Insulin)", "type": "호르몬 주사제", "note": "1형 당뇨·중증 2형, 처방 필요"},
            {"name": "다파글리플로진 (SGLT2억제제)", "type": "신형 혈당강하제", "note": "심혈관 보호 효과 추가, 처방 필요"},
        ],
        "treatments": ["혈당 자가 모니터링 (하루 2회 이상)", "저당·저GI 식이 관리", "규칙적 유산소 운동 (주 150분)", "체중 감량 5~10%만으로도 혈당 개선", "정기 HbA1c 검사 (3개월마다)"],
        "folk_remedies": ["여주(비터멜론) — 혈당 강하 효과 연구됨", "계피 — 인슐린 민감성 개선 소규모 연구", "메타 섬유(차전자피) 식전 섭취 — 혈당 급등 억제"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "Hypertension": {
        "drugs": [
            {"name": "암로디핀 (Amlodipine)", "type": "칼슘채널차단제", "note": "노바스크 등, 처방 필요"},
            {"name": "로사르탄 (Losartan)", "type": "ARB계", "note": "코자 등, 신장 보호 효과, 처방 필요"},
            {"name": "히드로클로로티아지드", "type": "이뇨제", "note": "1차 치료 병합에 자주 사용, 처방 필요"},
            {"name": "메토프로롤 (Metoprolol)", "type": "베타차단제", "note": "심박수 조절, 처방 필요"},
        ],
        "treatments": ["저염식 (하루 5g 미만)", "DASH 식이요법", "규칙적 유산소 운동", "금연·절주", "체중 관리", "혈압 매일 기록 (가정혈압계 활용)"],
        "folk_remedies": ["마늘 섭취 (경미한 혈압 강하 연구됨)", "비트 주스 (NO 생성 촉진)", "히비스커스 차 (소규모 연구에서 혈압 강하)"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "Migraine": {
        "drugs": [
            {"name": "이부프로펜 (Ibuprofen)", "type": "NSAIDs", "note": "초기 발작 시 효과적, 약국 구매"},
            {"name": "수마트립탄 (Sumatriptan)", "type": "트립탄계", "note": "편두통 특이적 치료, 처방 필요"},
            {"name": "나프록센 (Naproxen)", "type": "NSAIDs", "note": "지속 시간 긴 진통제, 약국 구매"},
            {"name": "메토클로프라미드", "type": "구토억제제", "note": "구역감 동반 시 병용, 처방 필요"},
        ],
        "treatments": ["어둡고 조용한 환경에서 휴식", "냉찜질/온찜질 (개인 반응 따라)", "편두통 유발 음식 파악·회피 (치즈, 레드와인, 초콜릿 등)", "규칙적 수면 패턴 유지", "두통 일지 작성"],
        "folk_remedies": ["마그네슘 보충제 (예방 효과 연구됨)", "리보플라빈(B2) 고용량 — 예방에 도움", "페버퓨 허브 — 전통적 편두통 예방"],
        "urgency": "경과 관찰",
        "urgency_color": "green",
    },
    "Common Cold": {
        "drugs": [
            {"name": "아세트아미노펜 (Acetaminophen)", "type": "해열진통제", "note": "타이레놀 등, 발열·통증 완화"},
            {"name": "슈도에페드린", "type": "충혈완화제", "note": "코막힘 완화, 약국 구매"},
            {"name": "덱스트로메토르판", "type": "진해제", "note": "기침 억제, 약국 구매"},
            {"name": "식염수 비강 스프레이", "type": "비강 세척", "note": "약국 구매, 부작용 없음"},
        ],
        "treatments": ["충분한 수분 섭취 (하루 2L 이상)", "충분한 휴식", "따뜻한 환경 유지", "가습기 사용", "손 자주 씻기 (전파 예방)"],
        "folk_remedies": ["꿀+생강+레몬 차", "닭고기 수프 (항염 효과 연구됨)", "아연 로젠지 (초기 복용 시 기간 단축 연구)", "에키네이셔 (면역 보조, 근거 논란 있음)", "증기 흡입 (코막힘 완화)"],
        "urgency": "경과 관찰",
        "urgency_color": "green",
    },
    "Pneumonia": {
        "drugs": [
            {"name": "아목시실린 (Amoxicillin)", "type": "항생제 (세균성 폐렴)", "note": "지역사회 획득 폐렴 1차, 처방 필요"},
            {"name": "아지트로마이신 (Azithromycin)", "type": "항생제 (비정형)", "note": "지트로맥스 등, 처방 필요"},
            {"name": "아세트아미노펜", "type": "해열진통제", "note": "발열·흉통 완화"},
            {"name": "기관지 확장제", "type": "흡입제", "note": "호흡 곤란 완화, 처방 필요"},
        ],
        "treatments": ["즉시 병원 방문 (폐렴은 자가치료 위험)", "충분한 수분 보충", "반좌위 자세로 호흡 편안하게", "산소포화도 모니터링", "폐렴구균 백신 예방접종 권장"],
        "folk_remedies": ["따뜻한 증기 흡입으로 가래 배출 도움", "꿀+생강차 보조 (항균 효과)", "프로바이오틱스 (면역 보조)"],
        "urgency": "즉시 병원",
        "urgency_color": "red",
    },
    "Bronchial Asthma": {
        "drugs": [
            {"name": "살부타몰 흡입제 (Salbutamol)", "type": "속효성 기관지확장제", "note": "벤토린 등, 발작 즉각 완화, 처방 필요"},
            {"name": "플루티카손 흡입제", "type": "흡입 스테로이드 (ICS)", "note": "장기 조절용, 처방 필요"},
            {"name": "몬테루카스트 (Montelukast)", "type": "류코트리엔 길항제", "note": "싱귤레어 등, 예방용, 처방 필요"},
        ],
        "treatments": ["알레르겐·자극물질 회피", "흡입기 올바른 사용법 교육", "천식 일지 작성", "심한 운동 전 예방 흡입제 사용", "독감 예방접종 매년 권장"],
        "folk_remedies": ["생강차 (기관지 항염 효과)", "강황 우유 (쿠르쿠민 항염)", "꿀+검은씨 (Nigella sativa) — 전통 처방"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "Gastroenteritis": {
        "drugs": [
            {"name": "경구수액제 (ORS)", "type": "수분 전해질 보충", "note": "포카리 등, 가장 중요한 1차 치료"},
            {"name": "로페라미드 (Loperamide)", "type": "지사제", "note": "이모디움 등, 약국 구매, 감염성은 주의"},
            {"name": "메토클로프라미드", "type": "구토억제제", "note": "구역감 심할 때, 처방 필요"},
            {"name": "프로바이오틱스", "type": "장내균총 회복", "note": "락토바실러스 등, 약국 구매"},
        ],
        "treatments": ["충분한 수분·전해질 보충이 핵심", "BRAT 식이 (바나나, 쌀, 사과소스, 토스트)", "기름지고 자극적인 음식 회피", "증상 48시간 이상 지속 시 병원 방문"],
        "folk_remedies": ["생강차 (구역 완화)", "매실청 희석액 (장 진정)", "흰쌀 죽 섭취", "민트차 (복부 경련 완화)"],
        "urgency": "경과 관찰",
        "urgency_color": "green",
    },
    "Jaundice": {
        "drugs": [
            {"name": "원인 치료 약물 (의사 처방)", "type": "원인에 따라 상이", "note": "황달 자체보다 원인 질환 치료가 핵심"},
            {"name": "우르소데옥시콜산 (UDCA)", "type": "담즙산", "note": "담즙 정체성 황달에 사용, 처방 필요"},
        ],
        "treatments": ["반드시 의사 진료 (황달은 원인 파악이 필수)", "충분한 수분 섭취", "알코올 완전 금지", "고지방 음식 회피"],
        "folk_remedies": ["민들레 차 (간 해독 보조 전통 처방)", "강황 차 (담즙 분비 촉진 전통)", "비트 주스 (간 기능 보조)"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "Malaria": {
        "drugs": [
            {"name": "클로로퀸 (Chloroquine)", "type": "항말라리아제", "note": "내성 없는 지역, 처방 필요"},
            {"name": "아르테미시닌 병합요법 (ACT)", "type": "항말라리아제", "note": "현재 표준 치료, 처방 필요"},
            {"name": "프리마퀸", "type": "재발 예방제", "note": "P.vivax 재발 방지, 처방 필요"},
        ],
        "treatments": ["즉시 병원 방문 필수", "모기 기피제·모기장 사용", "수분 보충", "해열 처치"],
        "folk_remedies": ["아르테미시아 쑥 차 (아르테미시닌 원료 식물 — 실제 효과 있으나 용량 불확실)", "키나 나무 껍질 (퀴닌의 원료)", "님 잎 — 전통 인도 의학에서 사용"],
        "urgency": "즉시 병원",
        "urgency_color": "red",
    },
    "Dengue": {
        "drugs": [
            {"name": "아세트아미노펜 (Acetaminophen)", "type": "해열진통제", "note": "타이레놀 — 이부프로펜·아스피린은 출혈 위험으로 금지"},
            {"name": "경구수액제 (ORS)", "type": "수분 보충", "note": "탈수 예방 필수"},
        ],
        "treatments": ["즉시 병원 방문 (혈소판 수치 모니터링 필수)", "충분한 휴식·수분 보충", "NSAIDs·아스피린 절대 복용 금지 (출혈 위험)", "혈소판 수치 정기 체크"],
        "folk_remedies": ["파파야 잎 추출물 (혈소판 증가 일부 연구)", "코코넛 워터 (전해질 보충)", "길로이 (Tinospora) 허브 — 인도 전통 처방"],
        "urgency": "즉시 병원",
        "urgency_color": "red",
    },
    "Typhoid": {
        "drugs": [
            {"name": "시프로플록사신 (Ciprofloxacin)", "type": "항생제", "note": "표준 치료, 처방 필요"},
            {"name": "아지트로마이신", "type": "항생제", "note": "소아·임산부, 처방 필요"},
            {"name": "세프트리악손", "type": "항생제 (주사)", "note": "중증 입원 치료, 처방 필요"},
        ],
        "treatments": ["즉시 병원 방문", "충분한 수분 보충", "위생적 음식·물 섭취", "장티푸스 예방접종 (여행 전 권장)"],
        "folk_remedies": ["바나나 (소화 부담 적음)", "쌀죽·연한 유동식", "꿀 희석액 (항균 보조)"],
        "urgency": "즉시 병원",
        "urgency_color": "red",
    },
    "Tuberculosis": {
        "drugs": [
            {"name": "HRZE 병합요법", "type": "항결핵제 표준 요법", "note": "이소니아지드+리팜피신+피라진아미드+에탐부톨, 처방 필요"},
            {"name": "6개월 이상 복약 필수", "type": "치료 기간", "note": "임의 중단 시 내성 결핵 발생 위험"},
        ],
        "treatments": ["즉시 병원 방문 (법정 전염병)", "격리 치료 (초기 2주)", "완전한 투약 순응 (DOT 권장)", "영양 보충 (단백질 충분히)", "가족 접촉자 검사"],
        "folk_remedies": ["마늘 (항균 보조)", "강황 (항염)", "홀리 바질(투시) — 인도 전통 처방 (보조적)"],
        "urgency": "즉시 병원",
        "urgency_color": "red",
    },
    "Heart attack": {
        "drugs": [
            {"name": "아스피린 (Aspirin) 300mg", "type": "혈소판 억제제", "note": "의심 즉시 씹어서 복용 (출혈 없을 때)"},
            {"name": "니트로글리세린 설하정", "type": "혈관확장제", "note": "처방 있을 때만, 혀 밑 녹임"},
            {"name": "헤파린·클로피도그렐", "type": "항혈전제", "note": "병원 도착 후 처방"},
        ],
        "treatments": ["119 즉시 호출", "누운 자세 유지", "꽉 끼는 옷 풀기", "CPR 준비", "AED 사용"],
        "folk_remedies": ["민간요법 시도 금지 — 즉각 119 신고가 생명을 구합니다"],
        "urgency": "즉시 병원",
        "urgency_color": "red",
    },
    "Hypothyroidism": {
        "drugs": [
            {"name": "레보티록신 (Levothyroxine)", "type": "갑상선 호르몬 보충제", "note": "씬지로이드 등, 처방 필요, 공복 복용"},
        ],
        "treatments": ["정기 TSH 혈액검사 (3~6개월마다)", "공복에 약 복용 (30분 후 식사)", "칼슘·철분 보충제와 시간 간격 두기", "규칙적 운동으로 피로감 개선"],
        "folk_remedies": ["셀레늄 보충 (갑상선 기능 보조 연구)", "아슈와간다 허브 (일부 연구)", "요오드 풍부 식품 (김, 미역) — 과도하면 역효과"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "Hyperthyroidism": {
        "drugs": [
            {"name": "메티마졸 (Methimazole)", "type": "항갑상선제", "note": "갑상선 호르몬 합성 억제, 처방 필요"},
            {"name": "프로프라놀롤 (Propranolol)", "type": "베타차단제", "note": "빠른 심박수·떨림 조절, 처방 필요"},
            {"name": "방사성 요오드 치료", "type": "비수술적 치료", "note": "병원에서 시행"},
        ],
        "treatments": ["정기 갑상선 기능 검사", "요오드 함유 식품·보충제 제한", "카페인 제한", "규칙적 모니터링"],
        "folk_remedies": ["레몬밤 차 (갑상선 자극 억제 전통 처방)", "버그위드 허브 — 항갑상선 효과 일부 연구", "브로콜리·배추 (갑상선 과잉 활성 억제 성분 포함 — 과도하면 역효과)"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "Hypoglycemia": {
        "drugs": [
            {"name": "포도당 15~20g 즉각 섭취", "type": "응급 처치", "note": "주스·사탕·설탕물 즉시 섭취"},
            {"name": "글루카곤 키트", "type": "응급 주사제", "note": "의식 없을 때, 처방 필요"},
            {"name": "디아족사이드", "type": "혈당 상승제", "note": "만성 저혈당, 처방 필요"},
        ],
        "treatments": ["15-15 규칙: 당 15g 섭취 → 15분 후 재측정", "규칙적 식사 시간 유지", "식사 거르지 않기", "혈당측정기 항상 휴대", "당뇨약 복용 중이면 반드시 의사 상담"],
        "folk_remedies": ["꿀 1~2 티스푼 즉각 섭취 (응급 당 공급)", "바나나 (당+칼륨 보충)", "오트밀 (지속적 혈당 유지)"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "Urinary tract infection": {
        "drugs": [
            {"name": "트리메토프림-설파메톡사졸", "type": "항생제", "note": "단순 UTI 표준, 처방 필요"},
            {"name": "니트로푸란토인", "type": "항생제", "note": "방광염 특이적, 처방 필요"},
            {"name": "페나조피리딘", "type": "진통제 (요도)", "note": "배뇨 통증 완화, 소변 색 변화"},
        ],
        "treatments": ["충분한 수분 섭취 (하루 2L 이상)", "앞→뒤 방향으로 회음부 닦기", "면 소재 속옷 착용", "성교 후 즉시 배뇨", "카페인·알코올 제한"],
        "folk_remedies": ["크랜베리 주스 (PAC 성분의 세균 부착 억제)", "D-만노스 보충제 (E.coli 부착 억제 연구)", "프로바이오틱스 (질 내 정상균총 회복)"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "Acne": {
        "drugs": [
            {"name": "벤조일퍼옥사이드", "type": "국소 항균제", "note": "약국 구매, 2.5~5% 제품"},
            {"name": "트레티노인 (레티노이드)", "type": "국소 비타민A 유도체", "note": "모공 각화 억제, 처방 필요"},
            {"name": "독시사이클린", "type": "경구 항생제", "note": "중증 여드름, 처방 필요"},
            {"name": "이소트레티노인", "type": "경구 레티노이드", "note": "결절성 여드름, 처방 필요, 임신 금기"},
        ],
        "treatments": ["하루 2회 순한 클렌저로 세안", "손으로 짜지 않기", "비코메도제닉 제품 사용", "자외선 차단제 사용", "침구류 자주 세탁"],
        "folk_remedies": ["티트리 오일 국소 도포 (항균)", "알로에베라 젤 (항염)", "꿀 마스크 (항균)", "녹차 추출물 토너 (항산화·항염)"],
        "urgency": "경과 관찰",
        "urgency_color": "green",
    },
    "Psoriasis": {
        "drugs": [
            {"name": "코르티코스테로이드 크림", "type": "국소 스테로이드", "note": "경증~중등도, 처방 필요"},
            {"name": "칼시포트리올 (비타민D 유도체)", "type": "국소제", "note": "두피 건선에 효과적, 처방 필요"},
            {"name": "메토트렉세이트", "type": "면역억제제", "note": "중증, 처방 필요"},
            {"name": "생물학적 제제 (TNF억제제 등)", "type": "표적 치료제", "note": "중증 불응성, 처방 필요"},
        ],
        "treatments": ["자극 없는 순한 보습제 매일 사용", "목욕 후 즉시 보습", "스트레스 관리 (악화 유발)", "자외선 치료 (병원 광치료)", "알코올·흡연 제한"],
        "folk_remedies": ["알로에베라 젤 국소 도포 (항염)", "어성초 크림 (전통 처방)", "오트밀 목욕 (가려움 완화)", "오메가-3 보충 (항염 효과)"],
        "urgency": "경과 관찰",
        "urgency_color": "green",
    },
    "Osteoarthritis": {
        "drugs": [
            {"name": "아세트아미노펜", "type": "진통제 1차선택", "note": "타이레놀 등, 약국 구매"},
            {"name": "이부프로펜 (Ibuprofen)", "type": "NSAIDs", "note": "항염·진통, 위장 보호제 병용 권장"},
            {"name": "글루코사민+콘드로이친", "type": "관절 보호제", "note": "연골 보호 효과 일부 연구, 약국 구매"},
            {"name": "관절 내 히알루론산 주사", "type": "관절강 주사", "note": "병원 시술"},
        ],
        "treatments": ["적정 체중 유지 (관절 부하 감소)", "저충격 운동 (수영·자전거)", "근력 강화 운동", "온열·냉찜질", "관절 보호대 사용"],
        "folk_remedies": ["생강·강황 섭취 (항염)", "유황 온천 입욕 (전통 처방)", "아보카도-소야 추출물 (연구됨)"],
        "urgency": "경과 관찰",
        "urgency_color": "green",
    },
    "Arthritis": {
        "drugs": [
            {"name": "이부프로펜 (Ibuprofen)", "type": "NSAIDs", "note": "항염·진통, 약국 구매"},
            {"name": "메토트렉세이트", "type": "DMARD (류마티스)", "note": "류마티스 관절염 1차, 처방 필요"},
            {"name": "프레드니솔론", "type": "스테로이드", "note": "급성 염증 조절, 처방 필요"},
        ],
        "treatments": ["적정 체중 유지", "규칙적 관절 운동", "물리치료", "온열·냉찜질 병용"],
        "folk_remedies": ["강황 + 흑후추 섭취 (항염)", "생강차", "오메가-3 어유 보충"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "(vertigo) Paroxysmal Positional Vertigo": {
        "drugs": [
            {"name": "메클리진 (Meclizine)", "type": "항히스타민제", "note": "어지럼 완화, 처방 필요"},
            {"name": "디멘히드리네이트", "type": "구토억제·어지럼 완화", "note": "드라마민, 약국 구매"},
        ],
        "treatments": ["엡리 이석 정복술 (Epley maneuver) — 효과 높음", "갑작스런 머리 움직임 피하기", "눕고 일어날 때 천천히", "이비인후과·신경과 방문"],
        "folk_remedies": ["생강차 (구역·어지럼 완화)", "은행 추출물 (혈액순환 보조 전통)", "충분한 수분 섭취"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "Chicken pox": {
        "drugs": [
            {"name": "아시클로버 (Acyclovir)", "type": "항바이러스제", "note": "발진 24시간 내 복용 효과적, 처방 필요"},
            {"name": "칼라민 로션", "type": "국소 진양제", "note": "약국 구매, 가려움 완화"},
            {"name": "아세트아미노펜", "type": "해열제", "note": "아스피린 금지 (라이 증후군 위험)"},
        ],
        "treatments": ["손톱 짧게 유지 (긁지 않기)", "헐렁한 면 소재 착용", "시원한 목욕", "학교·직장 격리 (수포 모두 딱지될 때까지)"],
        "folk_remedies": ["오트밀 목욕 (가려움 완화)", "베이킹소다 목욕", "알로에베라 젤 (냉각·진정)", "인디안 라일락(님) 잎 목욕 — 인도 전통"],
        "urgency": "경과 관찰",
        "urgency_color": "green",
    },
    "hepatitis A": {
        "drugs": [
            {"name": "지지 요법 (원인 치료제 없음)", "type": "대증 치료", "note": "충분한 휴식·수분·영양 보충"},
            {"name": "A형 간염 백신", "type": "예방 백신", "note": "노출 후 2주 내 접종 시 예방 가능"},
        ],
        "treatments": ["충분한 휴식", "고단백 저지방 식이", "알코올 완전 금지", "위생 철저 (분변-구강 경로)"],
        "folk_remedies": ["민들레 차 (간 해독 보조)", "밀크씨슬(실리마린) — 간 보호 연구됨", "강황 차"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "Hepatitis B": {
        "drugs": [
            {"name": "엔테카비르 (Entecavir)", "type": "항바이러스제", "note": "만성 B형 간염, 처방 필요"},
            {"name": "테노포비르 (Tenofovir)", "type": "항바이러스제", "note": "장기 치료, 처방 필요"},
            {"name": "인터페론 알파", "type": "면역조절제", "note": "일부 환자, 처방 필요"},
        ],
        "treatments": ["즉시 간 전문의 방문", "알코올 완전 금지", "B형 간염 백신 (미접종자)", "성관계 시 콘돔 사용", "정기 간기능·바이러스 검사"],
        "folk_remedies": ["밀크씨슬 (실리마린, 간 보호 연구됨)", "강황 (항염 보조)", "리코리스 뿌리 차 (전통 처방)"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "Hepatitis C": {
        "drugs": [
            {"name": "소발디+하보니 (DAA 요법)", "type": "직접작용 항바이러스제", "note": "완치율 95%+, 처방 필요"},
            {"name": "글레카프레비르+피브렌타스비르", "type": "DAA 병합", "note": "마비렛, 8주 치료, 처방 필요"},
        ],
        "treatments": ["즉시 간 전문의 방문 (완치 가능한 바이러스성 간염)", "알코올 완전 금지", "주사기 공유 절대 금지", "정기 간기능 검사"],
        "folk_remedies": ["밀크씨슬 (보조적)", "강황 (항염)", "민들레 차"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "Alcoholic hepatitis": {
        "drugs": [
            {"name": "프레드니솔론", "type": "스테로이드", "note": "중증 알코올성 간염, 처방 필요"},
            {"name": "아데메티오닌 (SAMe)", "type": "간 보호제", "note": "간세포 보호, 처방 필요"},
            {"name": "비타민 B군 (티아민 등)", "type": "영양 보충", "note": "결핍 교정, 약국 구매"},
        ],
        "treatments": ["완전 금주 (가장 중요한 치료)", "영양 보충 (고단백 식이)", "간 전문의 즉시 방문"],
        "folk_remedies": ["밀크씨슬 (간 보호 보조)", "민들레 차", "강황"],
        "urgency": "즉시 병원",
        "urgency_color": "red",
    },
    "Chronic cholestasis": {
        "drugs": [
            {"name": "우르소데옥시콜산 (UDCA)", "type": "담즙산", "note": "콜레스테롤 담즙산 조성 개선, 처방 필요"},
            {"name": "콜레스티라민", "type": "담즙산 결합제", "note": "가려움 완화, 처방 필요"},
        ],
        "treatments": ["간담도 전문의 방문", "지용성 비타민 보충 (A·D·E·K)", "저지방 식이", "알코올 금지"],
        "folk_remedies": ["민들레 차", "아티초크 차 (담즙 분비 촉진 전통)", "강황"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "Drug Reaction": {
        "drugs": [
            {"name": "원인 약물 즉시 중단", "type": "1차 처치", "note": "의심 약물 모두 중단 후 의사 상담"},
            {"name": "항히스타민제 (세티리진)", "type": "증상 완화", "note": "경증 피부 반응"},
            {"name": "에피네프린 자동주사기", "type": "응급 처치", "note": "아나필락시스 시 즉시 사용"},
        ],
        "treatments": ["원인 약물 식별·기록", "의무기록에 알레르기 등록", "아나필락시스 징후 시 119 즉시 호출"],
        "folk_remedies": ["냉찜질 (국소 피부 반응 완화)", "알로에베라 젤 (경증 피부 발진)"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "AIDS": {
        "drugs": [
            {"name": "항레트로바이러스 요법 (ART)", "type": "복합 항바이러스 요법", "note": "즉시 시작 권장, 처방 필요"},
            {"name": "비리어드+트루바다 등", "type": "ART 구성 약물", "note": "HIV 진료소 처방"},
        ],
        "treatments": ["즉시 감염내과 전문의 방문", "ART 복약 순응 (완치는 불가하나 정상 수명 가능)", "기회감염 예방 (폐렴구균·독감 백신)", "영양 관리·운동"],
        "folk_remedies": ["강황·아슈와간다 (면역 보조 전통 — 보조적 역할만)", "충분한 수면·스트레스 관리", "균형 잡힌 영양 섭취"],
        "urgency": "즉시 병원",
        "urgency_color": "red",
    },
    "Varicose veins": {
        "drugs": [
            {"name": "디오스민+헤스페리딘", "type": "정맥 강화제", "note": "다프론 등, 약국 구매"},
            {"name": "루토사이드", "type": "플라보노이드", "note": "혈관 투과성 감소, 약국 구매"},
            {"name": "경화요법 주사", "type": "시술", "note": "혈관외과 시술"},
        ],
        "treatments": ["의료용 압박 스타킹 착용", "장시간 서있기·앉아있기 회피", "다리 올리기 자세 (취침 시)", "규칙적 걷기 운동"],
        "folk_remedies": ["말밤나무 추출물 (에스신 성분, 정맥 강화 연구됨)", "포도씨 추출물 (혈관 보호)", "사과식초 국소 도포 (전통 처방)"],
        "urgency": "경과 관찰",
        "urgency_color": "green",
    },
    "Cervical spondylosis": {
        "drugs": [
            {"name": "이부프로펜 (Ibuprofen)", "type": "NSAIDs", "note": "항염·진통, 약국 구매"},
            {"name": "근이완제 (에페리손)", "type": "근육 이완제", "note": "근육 경직 완화, 처방 필요"},
            {"name": "가바펜틴", "type": "신경병증 통증제", "note": "방사통 동반 시, 처방 필요"},
        ],
        "treatments": ["물리치료·경추 운동", "자세 교정 (모니터 높이, 스마트폰 자세)", "경추 베개 사용", "온열 치료"],
        "folk_remedies": ["생강·강황 섭취 (항염)", "캡사이신 크림 국소 도포", "온찜질"],
        "urgency": "경과 관찰",
        "urgency_color": "green",
    },
    "Paralysis (brain hemorrhage)": {
        "drugs": [
            {"name": "즉시 병원 이송 — 약물 자가 복용 금지", "type": "응급", "note": "골든 타임 내 치료가 예후 결정"},
        ],
        "treatments": ["119 즉시 호출", "FAST 확인 (Face·Arm·Speech·Time)", "환자 안정 유지", "재활 치료 (발병 후 조기 시작)"],
        "folk_remedies": ["민간요법 시도 금지 — 즉각 119 신고"],
        "urgency": "즉시 병원",
        "urgency_color": "red",
    },
    "Dimorphic hemorrhoids(piles)": {
        "drugs": [
            {"name": "하이드로코르티손 좌약", "type": "국소 스테로이드", "note": "항문 염증·가려움, 약국 구매"},
            {"name": "리도카인 연고", "type": "국소 마취제", "note": "통증 완화, 약국 구매"},
            {"name": "차전자피 섬유", "type": "식이섬유 보충제", "note": "변비 개선, 약국 구매"},
        ],
        "treatments": ["고섬유 식이 (하루 25~35g)", "충분한 수분 섭취", "온수 좌욕 (하루 2~3회)", "배변 시 힘주기 최소화", "심한 경우 결찰술·수술 고려"],
        "folk_remedies": ["알로에베라 젤 국소 도포 (항염)", "위치하젤 패드 (수렴·항염)", "감자 슬라이스 냉찜질 (전통)"],
        "urgency": "경과 관찰",
        "urgency_color": "green",
    },
    "Hepatitis D": {
        "drugs": [
            {"name": "페그인터페론 알파", "type": "면역조절제", "note": "현재 유일한 치료 옵션, 처방 필요"},
            {"name": "부레브타이드 (Bulevirtide)", "type": "신약", "note": "유럽 승인 HDV 치료제, 처방 필요"},
        ],
        "treatments": ["간 전문의 즉시 방문", "B형 간염 예방이 D형 예방", "알코올 완전 금지"],
        "folk_remedies": ["밀크씨슬 (보조)", "강황"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "Hepatitis E": {
        "drugs": [
            {"name": "지지 요법 (원인 치료제 없음)", "type": "대증 치료", "note": "면역 정상인은 자연 회복"},
            {"name": "리바비린", "type": "항바이러스제", "note": "면역저하자·만성화 시, 처방 필요"},
        ],
        "treatments": ["충분한 휴식·영양", "알코올 금지", "임산부는 즉시 병원 (중증화 위험)"],
        "folk_remedies": ["민들레 차", "강황"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
    "Impetigo": {
        "drugs": [
            {"name": "무피로신 연고 (Mupirocin)", "type": "국소 항생제", "note": "박트로반 등, 처방 필요"},
            {"name": "세팔렉신 (Cephalexin)", "type": "경구 항생제", "note": "광범위 감염 시, 처방 필요"},
        ],
        "treatments": ["병변 청결 유지", "수건·침구 개인 사용", "학교·어린이집 완치 전 등원 금지", "손 자주 씻기"],
        "folk_remedies": ["꿀 국소 도포 (항균)", "알로에베라 젤", "강황 페이스트 (항균·항염 전통 처방)"],
        "urgency": "빠른 진료",
        "urgency_color": "orange",
    },
}

# ── 훈련 데이터 생성 ──────────────────────────────────────────────────────────
@st.cache_data
def build_training_data():
    all_syms = sorted(set(s for v in DISEASE_SYMPTOMS.values() for s in v))
    rows = []
    rng = np.random.default_rng(42)
    for disease, syms in DISEASE_SYMPTOMS.items():
        for _ in range(30):
            row = {s: 0 for s in all_syms}
            n = max(2, int(len(syms) * rng.uniform(0.65, 1.0)))
            for s in rng.choice(syms, size=min(n, len(syms)), replace=False):
                row[s] = 1
            noise_pool = [s for s in all_syms if s not in syms]
            if noise_pool:
                for ns in rng.choice(noise_pool, size=rng.integers(0, 3), replace=False):
                    row[ns] = 1
            row["disease"] = disease
            rows.append(row)
    df = pd.DataFrame(rows)
    return df, all_syms

@st.cache_resource
def train_models():
    df, all_syms = build_training_data()
    X, y = df[all_syms].values, df["disease"].values
    nb = GaussianNB(); nb.fit(X, y)
    rf = RandomForestClassifier(n_estimators=120, random_state=42); rf.fit(X, y)
    return nb, rf, all_syms

# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 증상 체크")
    st.caption("해당하는 증상을 모두 선택하세요")

    st.markdown("**데이터 출처**")
    st.markdown(
        "- [Kaggle kaushil268](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)\n"
        "- [Columbia DBMI KB](https://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html)\n"
        "- [HPO JAX.org](https://hpo.jax.org/data/annotations)"
    )
    st.divider()

    categories = {
        "전신 증상": ["fatigue","weight_loss","weight_gain","lethargy","malaise","restlessness","anxiety","mood_swings"],
        "발열·통증":  ["high_fever","mild_fever","headache","joint_pain","back_pain","stomach_pain","chest_pain","abdominal_pain","pain_behind_the_eyes"],
        "피부·외형":  ["itching","skin_rash","nodal_skin_eruptions","yellowish_skin","yellowing_of_eyes","dark_urine"],
        "소화기":     ["vomiting","nausea","indigestion","acidity","diarrhoea","constipation","loss_of_appetite"],
        "호흡기":     ["cough","breathlessness","phlegm","congestion","runny_nose","sinus_pressure","throat_irritation","continuous_sneezing"],
        "기타":       ["sweating","chills","shivering","dehydration","blurred_and_distorted_vision","fast_heart_rate","burning_micturition","weakness_in_limbs"],
    }

    selected_symptoms = []
    for cat, syms in categories.items():
        with st.expander(cat, expanded=(cat == "전신 증상")):
            for s in syms:
                if s in SYMPTOM_KR and st.checkbox(SYMPTOM_KR[s], key=s):
                    selected_symptoms.append(s)

    st.divider()
    top_n       = st.slider("상위 N개 질병 표시", 3, 15, 8)
    model_choice = st.radio("예측 모델", ["앙상블 (권장)", "Naive Bayes", "Random Forest"])

# ── 메인 제목 ─────────────────────────────────────────────────────────────────
st.title("🩺 증상 기반 질병·치료법 예측 대시보드")
st.caption("Kaggle Disease-Symptom Dataset (41 diseases) · ML 확률 예측 · 약품·치료법·민간요법 안내")

nb_model, rf_model, all_syms = train_models()

if not selected_symptoms:
    st.info("👈 왼쪽 사이드바에서 현재 증상을 선택하면 예측 결과가 나타납니다.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("지원 질병", "41종")
    c2.metric("분석 증상", "132개")
    c3.metric("ML 모델", "NB + RF")
    c4.metric("치료 DB", f"{len(TREATMENT_DB)}종")
    st.stop()

# ── 예측 ──────────────────────────────────────────────────────────────────────
input_vec = np.array([[1 if s in selected_symptoms else 0 for s in all_syms]])
nb_p  = dict(zip(nb_model.classes_, nb_model.predict_proba(input_vec)[0]))
rf_p  = dict(zip(rf_model.classes_, rf_model.predict_proba(input_vec)[0]))

result_rows = []
for d in DISEASE_SYMPTOMS:
    nb_val = nb_p.get(d, 0)
    rf_val = rf_p.get(d, 0)
    if model_choice == "앙상블 (권장)":
        prob = (nb_val + rf_val) / 2
    elif model_choice == "Naive Bayes":
        prob = nb_val
    else:
        prob = rf_val
    result_rows.append({"disease": d, "disease_kr": DISEASE_KR.get(d, d), "probability": prob})

result_df = (
    pd.DataFrame(result_rows)
    .sort_values("probability", ascending=False)
    .head(top_n)
    .reset_index(drop=True)
)
result_df["prob_pct"] = (result_df["probability"] * 100).round(1)

# ── 레이아웃: 예측 결과 ───────────────────────────────────────────────────────
st.subheader("🔍 예측 결과")
sym_str = " · ".join(SYMPTOM_KR.get(s, s) for s in selected_symptoms)
st.markdown(f"**선택된 증상 ({len(selected_symptoms)}개):** {sym_str}")

col_chart, col_cards = st.columns([3, 2])

with col_chart:
    colors = ["#E24B4A" if p >= 30 else "#BA7517" if p >= 15 else "#378ADD"
              for p in result_df["prob_pct"]]
    fig = go.Figure(go.Bar(
        x=result_df["prob_pct"],
        y=result_df["disease_kr"],
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in result_df["prob_pct"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>확률: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=max(300, top_n * 44),
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis_title="가능성 (%)",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        xaxis=dict(range=[0, min(100, result_df["prob_pct"].max() * 1.35)]),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "<span style='color:#E24B4A'>■</span> 높은 가능성 (≥30%)　"
        "<span style='color:#BA7517'>■</span> 중간 (15~30%)　"
        "<span style='color:#378ADD'>■</span> 낮음 (<15%)",
        unsafe_allow_html=True,
    )

with col_cards:
    for _, row in result_df.head(3).iterrows():
        p = row["prob_pct"]
        if p >= 30:
            bg, bd, tc = "#FCEBEB", "#E24B4A", "#A32D2D"
        elif p >= 15:
            bg, bd, tc = "#FAEEDA", "#BA7517", "#854F0B"
        else:
            bg, bd, tc = "#E6F1FB", "#378ADD", "#185FA5"
        st.markdown(
            f"<div style='background:{bg};border:1px solid {bd};border-radius:10px;"
            f"padding:.7rem 1rem;margin-bottom:8px;'>"
            f"<div style='font-size:15px;font-weight:500;color:{tc};'>{row['disease_kr']}</div>"
            f"<div style='font-size:12px;color:{tc};opacity:.75;'>{row['disease']}</div>"
            f"<div style='font-size:24px;font-weight:700;color:{tc};'>{p:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════════════════════
#  치료법 상세 섹션
# ════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("💊 질병별 치료법 상세 안내")
st.caption("상위 예측 질병의 약품·치료법·민간요법을 확인하세요. 반드시 전문가 상담 후 적용하세요.")

# 표시할 질병 선택 탭
top_diseases = result_df["disease"].tolist()
tab_labels   = [f"{DISEASE_KR.get(d,d)} ({result_df.loc[result_df['disease']==d,'prob_pct'].values[0]:.1f}%)"
                for d in top_diseases]
tabs = st.tabs(tab_labels)

for tab, disease in zip(tabs, top_diseases):
    with tab:
        info = TREATMENT_DB.get(disease)
        row  = result_df[result_df["disease"] == disease].iloc[0]

        # 긴급도 배지
        if info:
            urg = info["urgency"]
            urg_map = {"즉시 병원": ("🚨", "#FCEBEB", "#E24B4A", "#A32D2D"),
                       "빠른 진료": ("⚠️", "#FAEEDA", "#BA7517", "#854F0B"),
                       "경과 관찰": ("✅", "#EAF3DE", "#3B6D11", "#27500A")}
            icon, bg, bd, tc = urg_map.get(urg, ("ℹ️","#E6F1FB","#185FA5","#0C447C"))
            st.markdown(
                f"<div style='display:inline-block;background:{bg};border:1px solid {bd};"
                f"border-radius:8px;padding:4px 14px;font-size:13px;font-weight:500;"
                f"color:{tc};margin-bottom:12px;'>{icon} {urg}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("이 질병의 치료 정보는 데이터베이스에 추가 예정입니다.")
            continue

        col1, col2, col3 = st.columns(3)

        # ── 약품 ────────────────────────────────────────────────────────────
        with col1:
            st.markdown("### 💊 추천 약품")
            for drug in info["drugs"]:
                st.markdown(
                    f"<div style='background:var(--background-color);border:0.5px solid #b0b0b0;"
                    f"border-radius:8px;padding:.6rem .9rem;margin-bottom:8px;'>"
                    f"<div style='font-size:14px;font-weight:500;'>{drug['name']}</div>"
                    f"<div style='font-size:12px;color:#666;margin-top:2px;'>"
                    f"<span style='background:#E6F1FB;color:#185FA5;padding:1px 7px;"
                    f"border-radius:4px;font-size:11px;'>{drug['type']}</span></div>"
                    f"<div style='font-size:12px;color:#555;margin-top:4px;'>{drug['note']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── 치료법 ───────────────────────────────────────────────────────────
        with col2:
            st.markdown("### 🏥 치료·관리법")
            for i, t in enumerate(info["treatments"], 1):
                st.markdown(
                    f"<div style='display:flex;gap:8px;align-items:flex-start;"
                    f"margin-bottom:7px;'>"
                    f"<span style='background:#E1F5EE;color:#0F6E56;border-radius:50%;"
                    f"width:22px;height:22px;display:flex;align-items:center;justify-content:center;"
                    f"font-size:11px;font-weight:500;flex-shrink:0;'>{i}</span>"
                    f"<span style='font-size:13px;line-height:1.5;'>{t}</span></div>",
                    unsafe_allow_html=True,
                )

        # ── 민간요법 ─────────────────────────────────────────────────────────
        with col3:
            st.markdown("### 🌿 민간요법")
            st.caption("⚠️ 과학적 근거 수준이 다양합니다. 보조 수단으로만 활용하세요.")
            for remedy in info["folk_remedies"]:
                st.markdown(
                    f"<div style='display:flex;gap:8px;align-items:flex-start;"
                    f"margin-bottom:7px;'>"
                    f"<span style='color:#3B6D11;font-size:14px;flex-shrink:0;'>🌱</span>"
                    f"<span style='font-size:13px;line-height:1.5;'>{remedy}</span></div>",
                    unsafe_allow_html=True,
                )

# ── 전체 테이블 & 모델 정보 ───────────────────────────────────────────────────
with st.expander("📋 전체 예측 결과 테이블"):
    disp = result_df[["disease_kr","disease","prob_pct"]].copy()
    disp.columns = ["질병명 (한국어)", "질병명 (영어)", "가능성 (%)"]
    disp.index   = range(1, len(disp)+1)
    st.dataframe(disp, use_container_width=True)

with st.expander("🔧 데이터·모델 정보"):
    st.markdown("""
**데이터 출처**
| 데이터셋 | 링크 | 용도 |
|---|---|---|
| Kaggle kaushil268 | [링크](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) | 주 학습 (41 질병 × 132 증상) |
| itachi9604 GitHub | [링크](https://github.com/itachi9604/Disease-Symptom-dataset) | 원본 CSV + severity |
| Columbia DBMI KB | [링크](https://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html) | 임상 NLP 기반 보정 참고 |
| HPO JAX.org | [링크](https://hpo.jax.org/data/annotations) | 희귀질환 확장 참고 |

**모델**: Gaussian Naive Bayes + Random Forest (120 trees) 앙상블  
**한계**: 공개 데이터 기반으로 실제 임상 정확도와 차이가 있습니다. 반드시 의사 진료를 받으세요.
    """)
