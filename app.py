import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import io

# ==========================================
# THÀNH PHẦN 0: CẤU HÌNH TRANG & HÀM DÙNG CHUNG
# ==========================================
st.set_page_config(layout="wide", page_title="Hệ thống Phát hiện Gian lận", page_icon="🕵️")

@st.cache_data
def load_data(uploaded_file):
    """Hàm nạp dữ liệu dùng chung, có cache để tăng tốc độ"""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Định dạng file không được hỗ trợ!")
        return None
    return df

# Khởi tạo session_state nếu chưa có
if 'trained' not in st.session_state:
    st.session_state['trained'] = False
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = {}

# Đọc danh sách đặc trưng từ notebook
FEATURES = [f'X_{i}' for i in range(1, 15)]
TARGET = 'default'

# ==========================================
# THÀNH PHẦN 1: SIDEBAR - VÙNG CẤU HÌNH
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải dữ liệu
    uploaded_file = st.file_uploader("Tải lên dữ liệu huấn luyện (CSV/Excel)", type=['csv', 'xlsx'], help="Định dạng chứa các cột X_1 đến X_14 và cột default")
    
    # 2. Lựa chọn mô hình
    model_name = st.selectbox(
        "Lựa chọn thuật toán", 
        ["Random Forest", "Decision Tree", "Logistic Regression"],
        help="Thuật toán Machine Learning để phân loại giao dịch"
    )
    
    # 3. Tham số mô hình
    st.subheader("Tham số mô hình AI")
    with st.expander("Tùy chỉnh tham số", expanded=True):
        test_size = st.slider("Tỷ lệ tập kiểm thử (test_size)", 0.1, 0.5, 0.2, 0.05, help="Tỷ lệ dữ liệu dành cho việc đánh giá mô hình")
        random_state = st.number_input("Random State", value=32, step=1)
        
        if model_name == "Random Forest":
            n_estimators = st.slider("Số lượng cây (n_estimators)", 10, 500, 100, 10)
            max_depth = st.slider("Độ sâu tối đa (max_depth)", 2, 50, 10, 1)
        elif model_name == "Decision Tree":
            max_depth = st.slider("Độ sâu tối đa (max_depth)", 2, 50, 10, 1)
            criterion = st.selectbox("Tiêu chí phân nhánh", ["gini", "entropy"])
        elif model_name == "Logistic Regression":
            max_iter = st.slider("Số vòng lặp tối đa (max_iter)", 100, 2000, 1000, 100, help="Tăng giá trị này nếu thuật toán không hội tụ")
    
    st.divider()
    
    # Nút hành động
    train_btn = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# ==========================================
# THÀNH PHẦN 2: HEADER - VÙNG ĐỊNH HƯỚNG
# ==========================================
st.title("🕵️ Hệ thống Phát hiện Giao dịch Gian lận")
st.caption("Ứng dụng phân tích và dự báo rủi ro tín dụng / gian lận dựa trên Machine Learning. Áp dụng cho các dữ liệu giao dịch đã được mã hóa (X_1 đến X_14).")

if uploaded_file is None:
    st.info("👈 Vui lòng tải lên tập dữ liệu huấn luyện ở thanh bên trái (Sidebar) để bắt đầu.")
    st.stop()

# Đọc dữ liệu
df = load_data(uploaded_file)
if df is None:
    st.stop()

# Kiểm tra dữ liệu
missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]
if missing_cols:
    st.error(f"Dữ liệu tải lên thiếu các cột bắt buộc: {', '.join(missing_cols)}")
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}** | Số dòng: {df.shape[0]:,} | Số cột: {df.shape[1]}")
st.divider()

# ==========================================
# XỬ LÝ KHỐI TRAIN (Kích hoạt khi bấm nút)
# ==========================================
if train_btn:
    with st.spinner("Đang huấn luyện mô hình..."):
        # Phân chia dữ liệu
        X = df[FEATURES]
        y = df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Khởi tạo mô hình
        if model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)
        elif model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=max_iter, random_state=random_state)
            
        # Huấn luyện
        model.fit(X_train, y_train)
        
        # Dự báo và đánh giá
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Lưu vào session_state
        st.session_state['model'] = model
        st.session_state['metrics'] = {
            'cm': cm,
            'report': report,
            'model_name': model_name
        }
        st.session_state['trained'] = True
    st.success(f"Đã huấn luyện thành công mô hình {model_name}!")

# ==========================================
# KHỐI HIỂN THỊ CHÍNH (TABS)
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa", 
    "🎯 Kết quả kiểm định", 
    "🔮 Sử dụng mô hình"
])

# ------------------------------------------
# THÀNH PHẦN 3: TỔNG QUAN DỮ LIỆU
# ------------------------------------------
with tab1:
    st.subheader("Kích thước dữ liệu")
    col1, col2, col3 = st.columns(3)
    col1.metric("Số lượng bản ghi", f"{df.shape[0]:,}")
    col2.metric("Số lượng biến", f"{df.shape[1]}")
    file_size_mb = uploaded_file.size / (1024 * 1024)
    col3.metric("Dung lượng file", f"{file_size_mb:.2f} MB")
    
    st.subheader("Dữ liệu thô (5 dòng đầu)")
    with st.container(height=250):
        st.dataframe(df.head(), use_container_width=True)
        
    st.subheader("Thống kê mô tả các biến đặc trưng")
    st.dataframe(df[FEATURES + [TARGET]].describe(), use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 4: TRỰC QUAN HÓA DỮ LIỆU
# ------------------------------------------
with tab2:
    st.subheader("Phân phối dữ liệu")
    
    # Biểu đồ biến mục tiêu
    fig_target = px.bar(
        df[TARGET].value_counts().reset_index(), 
        x=TARGET, y='count', 
        labels={'count': 'Số lượng', TARGET: 'Nhãn (0: Bình thường, 1: Gian lận)'},
        title="Phân phối biến mục tiêu (default)",
        color=TARGET, color_discrete_sequence=['#2ecc71', '#ff4b4b']
    )
    st.plotly_chart(fig_target, use_container_width=True)
    
    st.divider()
    
    # Tự chọn biến để vẽ phân phối
    selected_features = st.multiselect(
        "Chọn các biến đầu vào để xem phân phối:", 
        FEATURES, 
        default=FEATURES[:4]
    )
    
    if selected_features:
        # Lưới 2 cột
        cols = st.columns(2)
        for i, feature in enumerate(selected_features):
            fig = px.histogram(
                df, x=feature, color=TARGET, 
                title=f"Phân phối {feature} theo nhãn",
                barmode='overlay',
                color_discrete_sequence=['#2ecc71', '#ff4b4b']
            )
            cols[i % 2].plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 5: KẾT QUẢ HUẤN LUYỆN
# ------------------------------------------
with tab3:
    if not st.session_state['trained']:
        st.info("Vui lòng thiết lập tham số và bấm nút **'🚀 Huấn luyện mô hình'** ở thanh bên trái để xem kết quả.")
    else:
        st.subheader(f"Hiệu suất mô hình: {st.session_state['metrics']['model_name']}")
        
        report = st.session_state['metrics']['report']
        cm = st.session_state['metrics']['cm']
        
        # Các chỉ số chính
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Độ chính xác (Accuracy)", f"{report['accuracy']:.2%}")
        # Chú ý: Gian lận thường là class '1'
        col2.metric("Precision (Gian lận - 1)", f"{report['1']['precision']:.2%}")
        col3.metric("Recall (Gian lận - 1)", f"{report['1']['recall']:.2%}")
        col4.metric("F1-Score (Gian lận - 1)", f"{report['1']['f1-score']:.2%}")
        
        st.divider()
        col_cm, col_rep = st.columns([1, 1])
        
        with col_cm:
            st.markdown("**Ma trận nhầm lẫn (Confusion Matrix)**")
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale='Blues',
                labels=dict(x="Nhãn Dự báo", y="Nhãn Thực tế", color="Số lượng"),
                x=['Bình thường (0)', 'Gian lận (1)'],
                y=['Bình thường (0)', 'Gian lận (1)']
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_rep:
            st.markdown("**Báo cáo phân loại (Classification Report)**")
            # Chuyển dict thành dataframe để hiển thị đẹp hơn
            df_report = pd.DataFrame(report).transpose().round(2)
            st.dataframe(df_report, use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 6: SỬ DỤNG MÔ HÌNH
# ------------------------------------------
with tab4:
    if not st.session_state['trained']:
        st.info("Vui lòng huấn luyện mô hình trước khi dự báo dữ liệu mới.")
    else:
        st.subheader("Sử dụng mô hình để dự báo")
        model = st.session_state['model']
        
        mode = st.radio("Chọn phương thức nhập liệu:", ["Nhập tay từng bản ghi", "Tải lên tệp dữ liệu (Hàng loạt)"], horizontal=True)
        
        if mode == "Nhập tay từng bản ghi":
            with st.form("predict_form"):
                st.write("Nhập thông số giao dịch (mặc định là giá trị trung vị của tập huấn luyện):")
                input_data = {}
                
                # Tạo lưới 3 cột để form không quá dài
                form_cols = st.columns(3)
                for i, feature in enumerate(FEATURES):
                    median_val = float(df[feature].median())
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    
                    with form_cols[i % 3]:
                        input_data[feature] = st.number_input(
                            feature, 
                            value=median_val,
                            format="%.6f"
                        )
                        
                submitted = st.form_submit_button("Dự báo", type="primary")
                
                if submitted:
                    input_df = pd.DataFrame([input_data])
                    pred = model.predict(input_df)[0]
                    prob = model.predict_proba(input_df)[0][1] # Xác suất gian lận
                    
                    st.divider()
                    if pred == 1:
                        st.error(f"🚨 **Cảnh báo:** Giao dịch có dấu hiệu GIAN LẬN! (Xác suất: {prob:.2%})")
                    else:
                        st.success(f"✅ Giao dịch BÌNH THƯỜNG. (Xác suất rủi ro: {prob:.2%})")
                        
        elif mode == "Tải lên tệp dữ liệu (Hàng loạt)":
            st.write("Tải lên tệp CSV/Excel chứa các giao dịch mới cần kiểm tra.")
            new_file = st.file_uploader("Tệp dữ liệu dự báo", type=['csv', 'xlsx'], key='new_data')
            
            if new_file is not None:
                new_df = load_data(new_file)
                missing_cols_new = [col for col in FEATURES if col not in new_df.columns]
                
                if missing_cols_new:
                    st.error(f"Tệp tải lên thiếu các biến bắt buộc: {', '.join(missing_cols_new)}")
                else:
                    X_new = new_df[FEATURES]
                    
                    if st.button("Bắt đầu chấm điểm (Scoring)"):
                        with st.spinner("Đang xử lý..."):
                            preds = model.predict(X_new)
                            probs = model.predict_proba(X_new)[:, 1]
                            
                            res_df = new_df.copy()
                            res_df['Prediction'] = preds
                            res_df['Risk_Probability'] = probs
                            
                        st.success("Dự báo hoàn tất!")
                        
                        # Hiển thị kết quả
                        st.dataframe(res_df.style.apply(
                            lambda x: ['background: #ffcccc' if v == 1 else '' for v in x], 
                            subset=['Prediction']
                        ), height=300)
                        
                        # Tải xuống
                        csv = res_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 Tải xuống kết quả (CSV)",
                            data=csv,
                            file_name="ket_qua_du_bao.csv",
                            mime="text/csv",
                        )
