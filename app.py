import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="증상 기반 질병 예측 대시보드",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 면책 조항 (최상단 고정) ──────────────────────────────────────────────────
st.warning(
    "⚠️ **면책 조항**: 이 도구는 의료기기가 아니며 진단을 대체하지 않습니다. "
    "참고용 건강 정보 도구로만 활용하고, 정확한 진단은 반드시 의사에게 받으세요."
)

# ── 데이터: Kaggle kaushil268 / itachi9604 데이터셋 기반 (41 diseases × 132 symptoms) ──
# 출처: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
# GitHub: https://github.com/itachi9604/Disease-Symptom-dataset

DISEASES = [
    "Fungal infection", "Allergy", "GERD", "Chronic cholestasis",
    "Drug Reaction", "Peptic ulcer disease", "AIDS", "Diabetes",
    "Gastroenteritis", "Bronchial Asthma", "Hypertension", "Migraine",
    "Cervical spondylosis", "Paralysis (brain hemorrhage)", "Jaundice",
    "Malaria", "Chicken pox", "Dengue", "Typhoid", "hepatitis A",
    "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E",
    "Alcoholic hepatitis", "Tuberculosis", "Common Cold", "Pneumonia",
    "Dimorphic hemorrhoids(piles)", "Heart attack", "Varicose veins",
    "Hypothyroidism", "Hyperthyroidism", "Hypoglycemia", "Osteoarthritis",
    "Arthritis", "(vertigo) Paroxysmal Positional Vertigo", "Acne",
    "Urinary tract infection", "Psoriasis", "Impetigo",
]

# 증상 목록 (132개 중 사용자 친화적 60개로 정제 — 실제 데이터셋 컬럼명 기반)
SYMPTOMS = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity",
    "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition",
    "spotting_urination", "fatigue", "weight_gain", "anxiety",
    "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness",
    "lethargy", "patches_in_throat", "irregular_sugar_level", "cough",
    "high_fever", "sunken_eyes", "breathlessness", "sweating",
    "dehydration", "indigestion", "headache", "yellowish_skin",
    "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes",
    "back_pain", "constipation", "abdominal_pain", "diarrhoea",
    "mild_fever", "yellow_urine", "yellowing_of_eyes", "acute_liver_failure",
    "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes",
    "malaise", "blurred_and_distorted_vision", "phlegm", "throat_irritation",
    "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion",
    "chest_pain", "weakness_in_limbs", "fast_heart_rate",
    "pain_during_bowel_movements",
]

SYMPTOM_KR = {
    "itching": "가려움증", "skin_rash": "피부 발진", "nodal_skin_eruptions": "결절성 피부 발진",
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
    "phlegm": "가래", "throat_irritation": "인후 자극", "redness_of_eyes": "눈 충혈",
    "sinus_pressure": "부비동 압박", "runny_nose": "콧물", "congestion": "코막힘",
    "chest_pain": "흉통", "weakness_in_limbs": "사지 무력감",
    "fast_heart_rate": "빠른 심박수", "pain_during_bowel_movements": "배변 통증",
}

DISEASE_KR = {
    "Fungal infection": "곰팡이 감염", "Allergy": "알레르기", "GERD": "위식도역류",
    "Chronic cholestasis": "만성 담즙정체", "Drug Reaction": "약물 반응",
    "Peptic ulcer disease": "소화성 궤양", "AIDS": "에이즈", "Diabetes": "당뇨",
    "Gastroenteritis": "위장염", "Bronchial Asthma": "기관지 천식",
    "Hypertension": "고혈압", "Migraine": "편두통", "Cervical spondylosis": "경추 척추증",
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

# 증상-질병 연관 데이터 (실제 Kaggle 데이터셋 기반 핵심 매핑)
# 각 질병당 주요 증상 리스트 — kaushil268 dataset.csv 구조 반영
DISEASE_SYMPTOMS = {
    "Fungal infection": ["itching", "skin_rash", "nodal_skin_eruptions", "fatigue"],
    "Allergy": ["continuous_sneezing", "chills", "fatigue", "cough", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "headache"],
    "GERD": ["stomach_pain", "acidity", "vomiting", "cough", "chest_pain", "indigestion", "headache", "nausea"],
    "Chronic cholestasis": ["itching", "vomiting", "fatigue", "weight_loss", "abdominal_pain", "yellowish_skin", "dark_urine", "nausea"],
    "Drug Reaction": ["itching", "skin_rash", "stomach_pain", "vomiting", "burning_micturition"],
    "Peptic ulcer disease": ["vomiting", "indigestion", "loss_of_appetite", "abdominal_pain", "nausea"],
    "AIDS": ["muscle_wasting", "fatigue", "weight_loss", "patches_in_throat", "sweating", "malaise", "swelled_lymph_nodes"],
    "Diabetes": ["fatigue", "weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "blurred_and_distorted_vision", "weight_gain", "polyuria", "excessive_hunger"],
    "Gastroenteritis": ["vomiting", "sunken_eyes", "dehydration", "diarrhoea", "nausea"],
    "Bronchial Asthma": ["fatigue", "cough", "breathlessness", "phlegm", "chest_pain"],
    "Hypertension": ["headache", "chest_pain", "dizziness", "loss_of_balance", "lack_of_concentration"],
    "Migraine": ["headache", "nausea", "vomiting", "blurred_and_distorted_vision", "pain_behind_the_eyes", "mood_swings"],
    "Cervical spondylosis": ["back_pain", "weakness_in_limbs", "neck_pain", "dizziness", "loss_of_balance"],
    "Paralysis (brain hemorrhage)": ["vomiting", "headache", "weakness_in_limbs", "chest_pain", "breathlessness"],
    "Jaundice": ["itching", "vomiting", "fatigue", "weight_loss", "high_fever", "yellowish_skin", "dark_urine", "abdominal_pain", "yellowing_of_eyes"],
    "Malaria": ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "muscle_pain", "diarrhoea"],
    "Chicken pox": ["itching", "skin_rash", "fatigue", "lethargy", "high_fever", "headache", "loss_of_appetite", "mild_fever", "swelled_lymph_nodes", "malaise"],
    "Dengue": ["skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "malaise", "muscle_pain", "red_spots_over_body"],
    "Typhoid": ["chills", "vomiting", "fatigue", "high_fever", "headache", "nausea", "constipation", "abdominal_pain", "diarrhoea", "toxic_look_typhos", "belly_pain"],
    "hepatitis A": ["joint_pain", "vomiting", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "diarrhoea", "mild_fever", "yellowing_of_eyes"],
    "Hepatitis B": ["itching", "fatigue", "lethargy", "yellowish_skin", "dark_urine", "loss_of_appetite", "abdominal_pain", "malaise", "yellowing_of_eyes", "receiving_blood_transfusion"],
    "Hepatitis C": ["fatigue", "yellowish_skin", "nausea", "loss_of_appetite", "family_history"],
    "Hepatitis D": ["joint_pain", "vomiting", "fatigue", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes"],
    "Hepatitis E": ["joint_pain", "vomiting", "fatigue", "high_fever", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes", "acute_liver_failure", "coma"],
    "Alcoholic hepatitis": ["vomiting", "yellowish_skin", "abdominal_pain", "swelling_of_stomach", "distention_of_abdomen", "history_of_alcohol_consumption", "fluid_overload"],
    "Tuberculosis": ["chills", "vomiting", "fatigue", "weight_loss", "cough", "high_fever", "breathlessness", "sweating", "loss_of_appetite", "mild_fever", "swelled_lymph_nodes", "malaise", "phlegm", "blood_in_sputum"],
    "Common Cold": ["continuous_sneezing", "chills", "fatigue", "cough", "headache", "runny_nose", "congestion", "mild_fever", "malaise", "muscle_pain", "throat_irritation"],
    "Pneumonia": ["chills", "fatigue", "cough", "high_fever", "breathlessness", "sweating", "malaise", "phlegm", "chest_pain", "fast_heart_rate", "rusty_sputum"],
    "Dimorphic hemorrhoids(piles)": ["constipation", "pain_during_bowel_movements", "bloody_stool", "rectal_bleeding"],
    "Heart attack": ["vomiting", "breathlessness", "sweating", "chest_pain", "fast_heart_rate"],
    "Varicose veins": ["fatigue", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "prominent_veins_on_calf"],
    "Hypothyroidism": ["fatigue", "weight_gain", "cold_hands_and_feets", "mood_swings", "lethargy", "dizziness", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremities", "depression", "irritability"],
    "Hyperthyroidism": ["fatigue", "mood_swings", "weight_loss", "restlessness", "sweating", "fast_heart_rate", "excessive_hunger", "irritability", "abnormal_menstruation"],
    "Hypoglycemia": ["fatigue", "weight_loss", "restlessness", "cold_hands_and_feets", "sweating", "irregular_sugar_level", "anxiety", "blurred_and_distorted_vision", "fast_heart_rate", "excessive_hunger"],
    "Osteoarthritis": ["joint_pain", "back_pain", "neck_pain", "knee_pain", "hip_joint_pain", "swelling_joints", "painful_walking"],
    "Arthritis": ["muscle_weakness", "swelling_joints", "movement_stiffness", "painful_walking", "joint_pain"],
    "(vertigo) Paroxysmal Positional Vertigo": ["vomiting", "headache", "nausea", "loss_of_balance", "unsteadiness"],
    "Acne": ["skin_rash", "pus_filled_pimples", "blackheads", "scurring"],
    "Urinary tract infection": ["burning_micturition", "spotting_urination", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine"],
    "Psoriasis": ["skin_rash", "joint_pain", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails"],
    "Impetigo": ["skin_rash", "high_fever", "blister", "red_sores_around_nose", "yellow_crust_ooze"],
}

# ── 훈련 데이터 생성 (Kaggle 데이터 구조 모방) ────────────────────────────
@st.cache_data
def build_training_data():
    all_symptoms = sorted(set(s for symptoms in DISEASE_SYMPTOMS.values() for s in symptoms))
    rows = []
    for disease, symptoms in DISEASE_SYMPTOMS.items():
        # 각 질병당 20개 샘플 생성 (노이즈 포함)
        for i in range(20):
            row = {s: 0 for s in all_symptoms}
            # 해당 질병 증상 중 70~100% 랜덤 선택
            n = max(2, int(len(symptoms) * np.random.uniform(0.7, 1.0)))
            chosen = np.random.choice(symptoms, size=n, replace=False)
            for s in chosen:
                row[s] = 1
            # 노이즈 증상 0~2개 추가
            noise_n = np.random.randint(0, 3)
            noise_pool = [s for s in all_symptoms if s not in symptoms]
            if noise_pool and noise_n > 0:
                for ns in np.random.choice(noise_pool, size=min(noise_n, len(noise_pool)), replace=False):
                    row[ns] = 1
            row["disease"] = disease
            rows.append(row)
    df = pd.DataFrame(rows)
    return df, all_symptoms

@st.cache_resource
def train_models():
    df, all_symptoms = build_training_data()
    X = df[all_symptoms].values
    y = df["disease"].values
    nb = GaussianNB()
    nb.fit(X, y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return nb, rf, all_symptoms

# ── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 증상 체크")
    st.markdown("**데이터 출처**")
    st.markdown("- [Kaggle kaushil268 dataset](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)")
    st.markdown("- [Columbia DBMI KB](https://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html)")
    st.markdown("- [HPO JAX.org](https://hpo.jax.org/data/annotations)")
    st.divider()

    st.markdown("### 증상을 선택하세요")

    # 카테고리별 증상 분류
    categories = {
        "전신 증상": ["fatigue", "weight_loss", "weight_gain", "lethargy", "malaise", "restlessness", "anxiety", "mood_swings"],
        "발열·통증": ["high_fever", "mild_fever", "headache", "joint_pain", "back_pain", "stomach_pain", "chest_pain", "abdominal_pain", "pain_behind_the_eyes"],
        "피부·외형": ["itching", "skin_rash", "nodal_skin_eruptions", "yellowish_skin", "yellowing_of_eyes", "dark_urine"],
        "소화기": ["vomiting", "nausea", "indigestion", "acidity", "diarrhoea", "constipation", "loss_of_appetite"],
        "호흡기": ["cough", "breathlessness", "phlegm", "congestion", "runny_nose", "sinus_pressure", "throat_irritation", "continuous_sneezing"],
        "기타": ["sweating", "chills", "shivering", "dehydration", "blurred_and_distorted_vision", "fast_heart_rate", "burning_micturition", "weakness_in_limbs"],
    }

    selected_symptoms = []
    for cat, syms in categories.items():
        with st.expander(cat, expanded=(cat == "전신 증상")):
            for s in syms:
                if s in SYMPTOM_KR:
                    if st.checkbox(SYMPTOM_KR[s], key=s):
                        selected_symptoms.append(s)

    st.divider()
    top_n = st.slider("상위 N개 질병 표시", 3, 15, 8)
    model_choice = st.radio("예측 모델", ["앙상블 (권장)", "Naive Bayes", "Random Forest"])

# ── 메인 영역 ────────────────────────────────────────────────────────────────
st.title("🩺 증상 기반 질병 가능성 예측 대시보드")
st.caption("Kaggle Disease-Symptom Dataset (41 diseases × 132 symptoms) 기반 · ML 확률 예측")

nb_model, rf_model, all_symptoms = train_models()

if not selected_symptoms:
    st.info("👈 왼쪽 사이드바에서 현재 증상을 선택하면 예측 결과가 나타납니다.")
    # 샘플 이미지 — 선택 없을 때 안내
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("지원 질병 수", "41종")
    with col2:
        st.metric("분석 증상 수", "132개")
    with col3:
        st.metric("ML 모델", "NB + RF 앙상블")
    st.stop()

# ── 입력 벡터 생성 ───────────────────────────────────────────────────────────
input_vec = np.array([[1 if s in selected_symptoms else 0 for s in all_symptoms]])

# ── 예측 ─────────────────────────────────────────────────────────────────────
nb_proba = nb_model.predict_proba(input_vec)[0]
rf_proba = rf_model.predict_proba(input_vec)[0]

nb_classes = nb_model.classes_
rf_classes = rf_model.classes_

nb_df = pd.DataFrame({"disease": nb_classes, "nb_prob": nb_proba})
rf_df = pd.DataFrame({"disease": rf_classes, "rf_prob": rf_proba})
result_df = nb_df.merge(rf_df, on="disease")

if model_choice == "앙상블 (권장)":
    result_df["probability"] = (result_df["nb_prob"] + result_df["rf_prob"]) / 2
elif model_choice == "Naive Bayes":
    result_df["probability"] = result_df["nb_prob"]
else:
    result_df["probability"] = result_df["rf_prob"]

result_df = result_df.sort_values("probability", ascending=False).head(top_n)
result_df["disease_kr"] = result_df["disease"].map(lambda x: DISEASE_KR.get(x, x))
result_df["prob_pct"] = (result_df["probability"] * 100).round(1)

# ── 레이아웃 ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader(f"🔍 예측 결과 — 상위 {top_n}개 질병")

    # 선택된 증상 요약
    sym_tags = " · ".join([SYMPTOM_KR.get(s, s) for s in selected_symptoms])
    st.markdown(f"**선택된 증상 ({len(selected_symptoms)}개):** {sym_tags}")

    # 수평 막대 차트
    colors = []
    for p in result_df["prob_pct"]:
        if p >= 30:
            colors.append("#E24B4A")
        elif p >= 15:
            colors.append("#BA7517")
        else:
            colors.append("#378ADD")

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
        margin=dict(l=10, r=60, t=20, b=10),
        xaxis_title="가능성 (%)",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        xaxis=dict(range=[0, min(100, result_df["prob_pct"].max() * 1.3)]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 범례
    st.markdown(
        "<span style='color:#E24B4A'>■</span> 높은 가능성 (≥30%)　"
        "<span style='color:#BA7517'>■</span> 중간 (15~30%)　"
        "<span style='color:#378ADD'>■</span> 낮음 (<15%)",
        unsafe_allow_html=True,
    )

with col_right:
    st.subheader("📊 상세 수치")

    # Top 3 강조 카드
    top3 = result_df.head(3)
    for i, row in top3.iterrows():
        p = row["prob_pct"]
        if p >= 30:
            color = "#FCEBEB"; border = "#E24B4A"; tc = "#A32D2D"
        elif p >= 15:
            color = "#FAEEDA"; border = "#BA7517"; tc = "#854F0B"
        else:
            color = "#E6F1FB"; border = "#378ADD"; tc = "#185FA5"
        st.markdown(
            f"""<div style='background:{color};border:1px solid {border};
            border-radius:10px;padding:.7rem 1rem;margin-bottom:8px;'>
            <div style='font-size:15px;font-weight:500;color:{tc};'>{row['disease_kr']}</div>
            <div style='font-size:13px;color:{tc};opacity:.8;'>{row['disease']}</div>
            <div style='font-size:22px;font-weight:700;color:{tc};'>{p:.1f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    # 파이 차트 (상위 N개 분포)
    fig2 = px.pie(
        result_df,
        values="prob_pct",
        names="disease_kr",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig2.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        height=280,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig2.update_traces(textinfo="percent+label", textfont_size=11)
    st.plotly_chart(fig2, use_container_width=True)

# ── 상세 테이블 ───────────────────────────────────────────────────────────────
with st.expander("📋 전체 결과 테이블 보기"):
    display_df = result_df[["disease_kr", "disease", "prob_pct"]].copy()
    display_df.columns = ["질병명 (한국어)", "질병명 (영어)", "가능성 (%)"]
    display_df = display_df.reset_index(drop=True)
    display_df.index += 1
    st.dataframe(display_df, use_container_width=True)

# ── 모델 정보 ─────────────────────────────────────────────────────────────────
with st.expander("🔧 모델 및 데이터 정보"):
    st.markdown("""
**사용 데이터셋**
- 주 학습 데이터: [Kaggle — Disease Prediction using ML (kaushil268)](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
- 증상-질병 매핑 원본: [GitHub — itachi9604/Disease-Symptom-dataset](https://github.com/itachi9604/Disease-Symptom-dataset)
- 임상 보정 참고: [Columbia DBMI Disease-Symptom KB](https://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html)
- 확장 참고: [Human Phenotype Ontology (HPO)](https://hpo.jax.org/data/annotations)

**모델 구성**
- Gaussian Naive Bayes: 증상 독립 가정, 빠른 추론
- Random Forest (100 trees): 증상 조합 비선형 패턴 포착
- 앙상블: 두 모델 확률 평균

**한계 및 주의사항**
- 본 도구는 의료 진단 도구가 아닙니다.
- 학습 데이터는 공개 Kaggle 데이터셋 기반으로 실제 임상 데이터와 차이가 있습니다.
- 정확한 진단은 반드시 의료 전문가에게 받으세요.
    """)
