import json
import pickle
import math

import pandas as pd
import streamlit as st
import random
from datetime import datetime
from modules.model_gensim import ModelGensim
# from modules.model_als import mode_als

# 
def display_product(result):
    idx = 0
    for _ in range(math.ceil(result.shape[0]/4)):
        cols = st.columns(4)
        i = 0 
        while idx < result.shape[0] and i < 4:
            with cols[i]:
                ele = result.iloc[idx]
                name = ele["name"] if len(ele["name"]) <= 80 else ele["name"][:80] + '...'
                st.write(name)
                st.image(ele["image"])
                st.write(str(ele['price']) + "đ")
                st.write('⭐'*int(ele['rating']))
            idx += 1
            i+= 1
    return None
# ------------ Load Mode ------------ # 
# --- Content Base --- #
@st.cache_resource
def upload():
    with open('checkpoint/model_gensim.pkl', 'rb') as inp:
        model_gensim = pickle.load(inp)
    path_gensim = 'data/ProductRaw.csv'
    df_gensim = pd.read_csv(path_gensim)
    df_gensim = df_gensim.dropna()
    # -------------------- #
    # --- Collaboration --- #
    # dataframe
    path_als = 'data/ReviewRaw.csv'
    df_als = pd.read_csv(path_als)[["customer_id", "product_id", "name", "rating"]].dropna()
    # model 
    json_file_path = "checkpoint/mode_als.json"
    with open(json_file_path, 'r') as j:
        dict_als = json.loads(j.read())
    return model_gensim, dict_als, df_gensim, df_als

model_gensim, dict_als, df_gensim, df_als = upload()
# --------------------- #
# ----------------------------------- # 

# ------------ GUI ------------ #
# menu
menu = ["Project Overview", "Build Project", "Content-base", "Collaboration"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Project Overview':    
    st.subheader("Project Overview")
    st.markdown("""---""")
    ### 1
    st.write(""" ## I. Project Objectives""")  
    st.write(""" """)
    ### 2
    st.write(""" ## II. Overview""")
    st.write(""" Recommender system là một kỹ thuật của trí tuệ nhân tạo, được nghiên cứu để cung cấp những gợi ý tự động tới người dùng hoặc khách hàng, dựa trên dữ liệu về hành vi trong quá khứ của họ. Recommender system có thể giúp người dùng tìm kiếm những sản phẩm, dịch vụ, nội dung hoặc đối tượng nào đó mà họ có thể quan tâm, thích thú hoặc cần thiết. Recommender system được ứng dụng rộng rãi trong nhiều lĩnh vực, như thương mại điện tử, giải trí, mạng xã hội, giáo dục và nhiều lĩnh vực khác""")
    st.write(""" Có nhiều phương pháp để xây dựng recommender system, nhưng chúng thường được chia thành ba loại chính, là: """)
    st.image("project.png", caption='', width= 700)
    st.write(""" - Content-based filtering: Phương pháp này dựa trên ý tưởng rằng những sản phẩm có nội dung hay thuộc tính tương tự sẽ được đánh giá hoặc ưa thích bởi cùng một người dùng. Phương pháp này sử dụng các kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) hoặc học máy (ML) để trích xuất các đặc trưng của sản phẩm, và so sánh chúng với sở thích của người dùng để gợi ý cho người dùng """)
    st.write(""" - Collaborative filtering: Phương pháp này dựa trên ý tưởng rằng những người dùng có sở thích tương tự sẽ đánh giá hoặc ưa thích những sản phẩm tương tự. Phương pháp này sử dụng các ma trận đánh giá hoặc ưa thích của người dùng để tìm ra những người dùng giống nhau hoặc những sản phẩm giống nhau, và dựa trên đó để gợi ý cho người dùng""")
    ### 3
    st.write(""" ## III. Demo""")
    st.write("Với demo này được xây dựng cho cả 2 phương pháp là: Content Based và Collaborative Filtering")
    st.write(""" ### Content Based: """)
    st.write(""" - Sự dụng thư viện gensim để tạo mô hình dự đoán """)
    st.write(""" - Có hai cách để tìm kiếm sản phẩm: 1. Chọn sản phẩm từ danh sách, 2. Nhập mô tả sản phẩm """)
    st.write(""" ### Collaborative Filtering : """)
    st.write(""" - Sự dụng als của pyspark để tạo mô hình dự đoán """)
    st.write(""" - Với tập data cho trước kết quả khi sử dụng model trên là rmse = 1.14 là khá tốt có thể đưa vào sử dụng  """)
    st.write(""" - Chọn tên khách hàng muốn để dự đoán những sản phẩm mà khách hàng có thể thích """)
elif choice == 'Build Project':
    st.subheader("Custom Model for Recommendation System")
    st.markdown("""---""")
    col1, col2 = st.columns([1,3])
    with col1:
        st.write(" ##### Select Model Train")
        train_type = st.radio(
            "",
            options=["Content Based", "Collaborative Filtering"],
        )
    with col2:
        # Upload file gensim
        if train_type == "Content Based":
            st.write(" ##### Summit a file for Content Based Model")
            uploaded_file_gensim = st.file_uploader("", type=['csv'])
            bt_start = st.button("Train")
            if uploaded_file_gensim and bt_start:
                # dataframe
                df_gensim = pd.read_csv(uploaded_file_gensim, encoding='latin-1').dropna()
                date = int(datetime.now().timestamp())
                df_gensim.to_csv(f"data/product_{date}.csv", index = False)
                df_train = df_gensim[['item_id', 'name', 'description', 'group']]
                df_train["name_view_group"] =  df_train['name'] + ' ' + df_train['description'] + ' ' + df_train['group']
                # model
                with st.spinner('Wait for it...'):
                    model_gensim_v1 = ModelGensim('data/vietnamese-stopwords.txt')
                    model_gensim_v1.train(df_train["name_view_group"].tolist())
                    # save_model
                    with open(f'checkpoint/model_gensim_{date}.pkl', 'wb') as fs:
                        pickle.dump(model_gensim_v1, fs, pickle.HIGHEST_PROTOCOL)
                model_gensim = model_gensim_v1
                st.success('Done!')

        # Upload file als
        elif train_type == "Collaborative Filtering":
            st.write("##### Summit a file for Collaborative Filtering Model")
            uploaded_file_als = st.file_uploader("", type=['csv'])
            bt_start = st.button("Train")
            if uploaded_file_als and bt_start:
                # with st.spinner('Wait for it...'):
                #     df_als = pd.read_csv(uploaded_file_als, encoding='latin-1')
                #     date = int(datetime.now().timestamp())
                #     df_als.to_csv(f"data/review{date}.csv", index = False)
                #     rmse = mode_als(df_als)
                #     st.write("Training: Done")
                #     st.write(f"Validation with RMSE = {rmse}")
                
                st.write('Done!')
        
elif choice == 'Content-base':
    # Upload file
    st.subheader("Content-Base")
    # paramenter
    col1, col2 = st.columns([1,3])
    with col1:
        search_type = st.radio(
            "Select Search Type",
            options=["Select", "Text"],
        )
    with col2:
        flat = True 
        text = ''
        with st.form("form_gensim"):
            if search_type == "Text":
                input_text = st.text_input('Text')
                text = input_text
                flat = False
            elif search_type == "Select":
                st.write("#### Product")
                input_text = st.selectbox('', df_gensim["name"].sample(20,random_state =42))
                # preprocess
                save = df_gensim[df_gensim['name'] == input_text].iloc[0]
                cont = st.container()
                cont.write(save["name"])
                cont.image(save["image"], width = 250)
                cont.write(str(save['price']) + "đ")
                cont.write('⭐'*int(save['rating']))
                text = save['name'] + save['description']
                # input_number 
            input_number = int( st.slider('Number', 0, 10, 5))
            clicked = st.form_submit_button("Search")
    # Search 
    if text and clicked:
        st.write("#### Similar product")
        output = model_gensim.predict(text,df_gensim)
        result = output.iloc[1:input_number+1] if flat else  output.iloc[:input_number]
        display_product(result)
            
elif choice == "Collaboration":   
    st.subheader("Collaboration")
    with st.form("form_gensim"):
        movie = random.sample(list(dict_als.keys()),20)
        movie = [int(x) for x in movie]
        df_search =  df_als[df_als["customer_id"].isin(movie)]['name']
        print(df_search)
        # preprocess
        st.write('#### Name Customer')
        input_text = st.selectbox('',df_search)
        save = df_als[df_als['name'] == input_text].iloc[0]
        id_cus = save['customer_id']
        input_number = int( st.slider('Number', 0, 10, 5))
        clicked = st.form_submit_button("Search")
    if id_cus and clicked:
        recom = dict_als.get(str(id_cus)) 
        st.write("### Product are recommended")
        if recom:
            recom = [x[0] for x in recom][:input_number+1]
            result = df_gensim[df_gensim['item_id'].isin(recom)]
            display_product(result)
        else:
            st.write("Not ID Customer in Dataset")
        
        # sanr phaamr da mua 
        st.markdown("""---""")
        st.write("### Product are purchased by customer")        
        list_pro = df_als[df_als["customer_id"] == id_cus]["product_id"].tolist()
        result2 = df_gensim[df_gensim['item_id'].isin(list_pro)]
        display_product(result2)
