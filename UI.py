import tkinter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tkinter import messagebox

window = tkinter.Tk()
window.title("와인예측")
window.geometry("300x330")

frame1 = tkinter.Frame(window,bg="white", relief='solid',bd=2)
frame1.pack(ipadx=100, ipady=2, padx=10,pady=10, side='top', fill='both')

tod = tkinter.Label(frame1, text="와인 품질 예측", bg="yellow",font=("나눔바른펜", 15))
tod.pack()

frame2 = tkinter.Frame(window,bg="white", relief='solid',bd=2)
frame2.pack(ipadx=100, ipady=60, padx=10,pady=10, side='top')

inp_volatile = tkinter.StringVar()
inp_sulphates = tkinter.StringVar()
inp_alcohol = tkinter.StringVar()

label = tkinter.Label(frame2, text="휘발성 산도를 입력!", font=("나눔바른펜", 13), bg="white")
label.pack(padx=20, pady=10)
textbox = tkinter.Entry(frame2, width=20, textvariable=inp_volatile)
textbox.pack()

label = tkinter.Label(frame2, text="황산염 수치를 입력!", font=("나눔바른펜", 13), bg="white")
label.pack(padx=20, pady=10)
textbox = tkinter.Entry(frame2, width=20, textvariable=inp_sulphates)
textbox.pack()

label = tkinter.Label(frame2, text="알코올 입력!", font=("나눔바른펜", 13), bg="white")
label.pack(padx=20, pady=10)
textbox = tkinter.Entry(frame2, width=20, textvariable=inp_alcohol)
textbox.pack()


def clickMe():
    volatile = float(inp_volatile.get())
    sulphates = float(inp_sulphates.get())
    alcohol = float(inp_alcohol.get())

    wineing = pd.read_csv('winequality-red.csv')

    # 트레인세트, 테스트세트 나누기
    train_set, test_set = train_test_split(wineing, test_size=0.2, random_state=42)

    train = train_set.drop("quality", axis=1)  # x_train
    train_labels = train_set["quality"].copy()  # y_train
    test = test_set.drop("quality", axis=1)  # x_test
    test_labels = test_set["quality"].copy()  # y_test

    # print('------------train_set로 학습,예측---------------')
    model = RandomForestClassifier(random_state=42)
    model.fit(train, train_labels)

    y_predict = model.predict(train)

    recommend = model.predict([[
        wineing["fixed acidity"].mean(),
        volatile,
        wineing["citric acid"].mean(),
        wineing["residual sugar"].mean(),
        wineing["chlorides"].mean(),
        wineing["free sulfur dioxide"].mean(),
        wineing["total sulfur dioxide"].mean(),
        wineing["density"].mean(),
        wineing["pH"].mean(),
        sulphates,
        alcohol]])

    messagebox.showinfo("와인 품질: ", recommend)


action = tkinter.Button(frame2, text="입력", command=clickMe,font=("나눔바른펜", 10), width=5)
action.pack()

window.mainloop()




