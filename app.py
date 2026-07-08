import streamlit as st
import pandas as pd
import pickle


# -------------------------
# 모델 불러오기
# -------------------------

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

le_airline = pickle.load(open("le_airline.pkl", "rb"))
le_origin = pickle.load(open("le_origin.pkl", "rb"))
le_dest = pickle.load(open("le_dest.pkl", "rb"))

features = pickle.load(open("features.pkl", "rb"))


# 정확도
accuracy = 0.9322


# -------------------------
# 화면 디자인
# -------------------------

st.set_page_config(
    page_title="항공기 지연 예측 AI",
    page_icon="✈️"
)


st.title("✈️ AI 항공기 지연 예측 시스템")

st.write(
    "머신러닝(Logistic Regression)을 활용하여 "
    "항공기 지연 가능성을 예측합니다."
)


st.divider()


# -------------------------
# 입력
# -------------------------

airline = st.selectbox(
    "✈️ 항공사",
    le_airline.classes_
)


origin = st.selectbox(
    "🛫 출발 공항",
    le_origin.classes_
)


dest = st.selectbox(
    "🛬 도착 공항",
    le_dest.classes_
)


dep_hour = st.slider(
    "출발 시간",
    0,
    23,
    12
)


month = st.selectbox(
    "월",
    list(range(1,13))
)


dayofweek = st.selectbox(
    "요일",
    [
        ("월요일",0),
        ("화요일",1),
        ("수요일",2),
        ("목요일",3),
        ("금요일",4),
        ("토요일",5),
        ("일요일",6)
    ],
    format_func=lambda x:x[0]
)


distance = st.number_input(
    "비행 거리(mile)",
    value=500
)


# -------------------------
# 예측
# -------------------------

if st.button("🚀 지연 예측하기"):

    airline_code = le_airline.transform([airline])[0]
    origin_code = le_origin.transform([origin])[0]
    dest_code = le_dest.transform([dest])[0]


    data = pd.DataFrame(
        [[
            airline_code,
            origin_code,
            dest_code,
            dep_hour,
            month,
            dayofweek[1],
            distance
        ]],
        columns=features
    )


    data_scaled = scaler.transform(data)


    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]


    st.divider()


    if pred == 1:

        st.error("🔴 지연 예상")

    else:

        st.success("🟢 정시 예상")


    st.metric(
        "지연 확률",
        f"{prob*100:.2f}%"
    )


    st.write("### 📊 모델 정보")

    st.write(
        f"테스트 정확도 : {accuracy*100:.2f}%"
    )


    st.info(
        """
        사용 알고리즘:
        Logistic Regression

        개발 환경:
        Python + Scikit-learn + Streamlit
        """
    )
