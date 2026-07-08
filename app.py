import gradio as gr
import pandas as pd
import pickle


# 모델 불러오기

model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

le_airline = pickle.load(open("le_airline.pkl","rb"))
le_origin = pickle.load(open("le_origin.pkl","rb"))
le_dest = pickle.load(open("le_dest.pkl","rb"))

features = pickle.load(open("features.pkl","rb"))


accuracy = 0.93


def predict_delay(
    airline,
    origin,
    dest,
    dep_hour,
    month,
    dayofweek,
    distance
):

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
            dayofweek,
            distance
        ]],
        columns=features
    )


    data = scaler.transform(data)


    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]


    if pred == 1:
        result="🔴 지연 예상"
    else:
        result="🟢 정시 예상"


    return f"""
# ✈️ 항공기 지연 예측 결과

## {result}

### 지연 확률
## {prob*100:.2f}%


---

### 입력 정보

항공사 : {airline}

출발공항 : {origin}

도착공항 : {dest}

출발시간 : {dep_hour}시

월 : {month}

요일 : {dayofweek}

거리 : {distance} mile


---

### 모델 정확도

{accuracy*100:.2f}%

"""


demo = gr.Interface(

    fn=predict_delay,

    inputs=[

        gr.Dropdown(
            choices=list(le_airline.classes_),
            label="항공사"
        ),

        gr.Dropdown(
            choices=list(le_origin.classes_),
            label="출발 공항"
        ),

        gr.Dropdown(
            choices=list(le_dest.classes_),
            label="도착 공항"
        ),

        gr.Slider(
            0,23,
            value=12,
            label="출발 시간"
        ),

        gr.Dropdown(
            list(range(1,13)),
            label="월"
        ),

        gr.Slider(
            0,6,
            value=0,
            label="요일"
        ),

        gr.Number(
            value=500,
            label="거리(mile)"
        )

    ],

    outputs="markdown",

    title="✈️ AI 항공기 지연 예측 시스템",

    description="Machine Learning 기반 항공기 지연 예측 서비스"

)


demo.launch()
