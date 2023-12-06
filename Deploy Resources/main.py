import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import pickle
# Menu
with st.sidebar:
    selected = option_menu("Dashboard", ["Informasi", "Prediksi Churn", "Segmentasi Pelanggan","Tentang Orbit Future Academy","Tim Kami"], icons=['search', 'plus', 'activity', 'info', 'users'], menu_icon="cast", default_index=1)

# Informasi
if selected == 'Informasi':
    st.title('FUTURE: Finding Ultimate Tactics to Reduce E-Commerce Churn Using Machine Learning Approach')
    info_title = "FUTURE: Finding Ultimate Tactics to Reduce E-Commerce Churn Using Machine Learning Approach"
    
    info_description = """
    <p style="text-align: justify;text-justify: inter-word;">
    Aplikasi ini dibuat untuk membantu bisnis e-commerce dalam 
    memprediksi pelanggan yang berpotensi berhenti berbelanja (churn) dan juga 
    melakukan segmentasi pelanggan dengan memanfaatkan data pelanggan yang ada.
    Harapannya, dapat memberikan wawasan yang berharga kepada bisnis untuk mengambil tindakan proaktif dalam 
    mempertahankan pelanggan yang berharga. 
    Dengan demikian, tujuan utama adalah mengurangi tingkat churn, meningkatkan retensi pelanggan, 
    dan mengoptimalkan strategi pemasaran serta pelayanan pelanggan untuk mencapai kesuksesan jangka panjang dalam industri e-commerce yang sangat kompetitif.</p>
    """
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(info_description, unsafe_allow_html=True)

    with col2:
        st.image("predik.png", width=400, caption="")

# Input
if selected == 'Prediksi Churn':
    st.title('Prediksi Churn')

    Complain_mapping = {"Ya": 1, "Tidak": 0}
    PreferedOrderCat_mapping = {
        "Laptop & Accessory": 2,
        "Mobile Phone": 3,
        "Grocery": 1,
        "Fashion": 0,
        "Other": 4
    }
    MaritalStatus_mapping = {"Single": 2, "Married": 1, "Divorced": 0}
    NumberOfDeviceRegistered_mapping = {
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6"
    }
    Complain = st.radio("Apakah pelanggan melakukan komplain di bulan sebelumnya?", list(Complain_mapping.keys()))
    PreferedOrderCat = st.radio("Kategori yang diminati pelanggan", list(PreferedOrderCat_mapping.keys()))
    MaritalStatus = st.radio("Status pernikahan", list(MaritalStatus_mapping.keys()))
    SatisfactionScore = st.slider("Nilai kepuasan pelanggan", min_value=1, max_value=5)
    NumberOfDeviceRegistered = st.number_input("Jumlah gawai yang tertaut", min_value=1, max_value=6, step=1)


    Complain_encoded = Complain_mapping[Complain]
    PreferedOrderCat_encoded = PreferedOrderCat_mapping[PreferedOrderCat]
    MaritalStatus_encoded = MaritalStatus_mapping[MaritalStatus]
    NumberOfDeviceRegistered_encoded = NumberOfDeviceRegistered_mapping[NumberOfDeviceRegistered]

    data = [
        [Complain_encoded, PreferedOrderCat_encoded, MaritalStatus_encoded, SatisfactionScore, NumberOfDeviceRegistered_encoded]
    ]

    if st.button("Prediksi"):
        # Load model dan skaler yang telah disimpan sebelumnya
        scaler = pickle.load(open('scaler_pred.pkl', 'rb'))
        best_model_classification = pickle.load(open('model_pred.pkl', 'rb'))

        # Standardisasi data
        data_scaled = scaler.transform(data)

        # Prediksi hasil Status
        hasil_prediksi = best_model_classification.predict(data_scaled)
        hasil_prediksi = int(hasil_prediksi)

        # Mapping hasil prediksi ke label yang sesuai
        if hasil_prediksi == 0:
            status = "Not Churn"
        else:
            status = "Churn"

        # Menampilkan hasil prediksi
        st.write(f"Hasil Prediksi: {status}")

# Customer Segmentation

if selected == 'Segmentasi Pelanggan':
    st.title('Segmentasi Pelanggan')
    OrderCount = st.number_input("Jumlah transaksi yang dilakukan pelanggan pada bulan sebelumnya", min_value=1, max_value=16, step=1)
    CouponUsed = st.number_input("Jumlah kupon yang digunakan pelanggan pada bulan sebelumnya", min_value=0, max_value=15, step=1)
    HourSpendOnApp = st.number_input("Banyaknya waktu penggunaan aplikasi (dalam jam)", min_value=0, max_value=5, step=1)
    
    # Input fitur
    data = [
        [
            OrderCount,
            CouponUsed,
            HourSpendOnApp
        ]
    ]

    if st.button("Kategorikan Pelanggan"):
        # Load model dan skaler yang telah disimpan sebelumnya
        MN = pickle.load(open('scaler_cusseg.pkl', 'rb'))
        best_model_cusseg = pickle.load(open('model_cusseg.pkl', 'rb'))  # Menggunakan best_model_cusseg

        # Standardisasi data
        data_scaled = MN.transform(data)

        # Prediksi hasil Customer Segmentation
        hasil_customer_segmentation = best_model_cusseg.predict(data_scaled)
        hasil_customer_segmentation = int(hasil_customer_segmentation)

        # Mapping hasil prediksi ke label yang sesuai
        if hasil_customer_segmentation == 0:
            status = "Pelanggan masuk di dalam cluster ke-0"
            penjelasan = "Cluster ke-0 adalah kelompok customer yang paling sedikit bertransaksi. Walaupun telah menghabiskan cukup banyak waktu browsing di aplikasi, \nkupon yang digunakan juga sangat sedikit. (Customer yang sangat pemilih dan sering membandingkan kualitas dan harga. \nTermasuk hanya bertransaksi barang-barang yang diperlukan saja walaupun tidak sedang ada kupon)"        
        elif hasil_customer_segmentation == 1:
            status = "Pelanggan masuk di dalam cluster ke-1"
            penjelasan = "Cluster ke-1 adalah kelompok customer yang sangat sering bertransaksi dan juga sangat sering menggunakan kupon. \nSelain itu, customer pun sering menghabiskan waktu untuk browsing aplikasi. \n(Customer yang mudah tergiur dengan adanya kupon yang sebisa mungkin diusahakan untuk digunakan)"
        elif hasil_customer_segmentation == 2:
            status = "Pelanggan masuk di dalam cluster ke-2"
            penjelasan = "Cluster ke-2 adalah kelompok customer yang bertransaksi cukup moderat dan juga sering menggunakan kupon, serta menghabiskan cukup banyak waktu. \nKemungkinan besar, setiap transaksi selalu menggunakan kupon. (Customer yang juga mudah tergiur dengan kupon, \nnamun setelah berpikir cukup panjang, tidak jadi bertransaksi)"
        elif hasil_customer_segmentation == 3:
            status = "Pelanggan masuk di dalam cluster ke-3"
            penjelasan = "Cluster ke-3 adalah kelompok customer yang paling sedikit bertansaksi, tidak pernah menggunakan kupon, dan menghabiskan sangat sedikit waktu \nuntuk browsing aplikasi. (Customer yang benar-benar hanya membuka aplikasi dan bertransaksi sesuai kebutuhan saja dan \nmungkin tidak tahu bagaimana cara menggunakan kupon atau tidak pernah eligible untuk mendapatkan kupon)"
        elif hasil_customer_segmentation == 4:
            status = "Pelanggan masuk di dalam cluster ke-4"
            penjelasan = "Cluster ke-4 adalah kelompok customer yang bertransaksi cukup moderat dan sangat sedikit menghabiskan \nwaktu untuk browsing di aplikasi. Namun, 50% transaksi selalu menggunakan kupon. \n(Customer yang selalu sudah memiliki rencana untuk membeli barang apa dan sangat sering mendapatkan kupon yang dapat dipakai)"
        elif hasil_customer_segmentation == 5:
            status = "Pelanggan masuk di dalam cluster ke-5"
            penjelasan = "Cluster ke-5 adalah kelompok customer yang bertransaksi cukup moderat dan selalu menggunakan kupon saat bertransaksi, \nnamun menghabiskan waktu paling lama browsing aplikasi. (Customer yang cukup mudah tergiur dengan kupon dan berusaha mencari barang-barang \ndengan value terbaik, namun sering pada akhirnya hanya cuci mata saja)"
        elif hasil_customer_segmentation == 6:
            status = "Pelanggan masuk di dalam cluster ke-6"
            penjelasan = "Cluster ke-6 adalah kelompok customer yang bertransaksi cukup moderat dan jarang menggunakan kupon, walaupun juga \nmenghabiskan waktu cukup lama browsing aplikasi. (Customer yang memiliki keinginan untuk membeli suatu barang spesifik \ndan melakukan riset panjang untuk menentukan keputusan baik dengan/tanpa kupon)"
        elif hasil_customer_segmentation == 7:
            status = "Pelanggan masuk di dalam cluster ke-7"
            penjelasan = "Cluster ke-7 adalah kelompok customer yang bertransaksi paling banyak dan menghabiskan banyak kupon juga. \nNamun, tidak terlalu lama browsing di aplikasi. (Customer yang sangat mudah tergiur dengan iklan/live streaming/harbolnas \nsehingga sering mendapatkan kupon. Namun, tidak terlalu sering membanding-bandingkan harga)"
        elif hasil_customer_segmentation == 8:
            status = "Pelanggan masuk di dalam cluster ke-8"
            penjelasan = "Cluster ke-8 adalah kelompok customer yang bertransaksi sangat sedikit dan menghabiskan sedikit waktu untuk browsing aplikasi. \nNamun, dapat dipastikan selalu menggunakan kupon. (Customer sudah memiliki barang spesifik yang ingin dibeli, \nnamun menunggu ada kupon untuk bertransaksi, tidak terlalu lama membanding-bandingkan harga lagi)"

        # Menampilkan hasil prediksi
        st.write(f"Kategori Pelanggan: {status}")
        st.write(f"Penjelasan Cluster: {penjelasan}")


 # About Us
if selected == 'Tentang Orbit Future Academy':
    st.title('Tentang Orbit Future Academy')
    aboutus_title = "Orbit Future Academy"
    
    aboutus_description = """
    <p style="text-align: justify;text-justify: inter-word;">Orbit Future Academy memberikan rangkaian program pelatihan, peningkatan, dan penyesuaian keterampilan di seluruh wilayah ASEAN, serta berencana untuk berkembang ke negara-negara lain di masa depan.
    Kami mengkhususkan diri dalam proyek besar B2B dan B2G di seluruh negeri untuk perusahaan, lembaga donor, dan pemerintah dalam program-program pelatihan digital dan Industri 4.0. Kami terus termotivasi untuk mencapai para pembelajar di berbagai geografi dan demografi, serta memberikan mereka keterampilan untuk mencapai hasil yang diinginkan, baik itu dalam mendapatkan pekerjaan atau menciptakan peluang kerja sebagai seorang pengusaha.
    Produk dan layanan kami yang telah banyak mendapatkan penghargaan dan diakui menggunakan teknologi untuk memberikan solusi yang dapat diperluas, tetapi selalu didukung dengan akses dan dukungan pendidik.</p>
    """
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(aboutus_description, unsafe_allow_html=True)

    with col2:
        st.image("Orbiit.png", width=300, caption="")
        st.markdown('[Link to Orbit Future Academy](https://orbitfutureacademy.id)', unsafe_allow_html=True)


# Our Team
if selected == 'Tim Kami':
    st.title('Tim Kami')
    st.write('Anggota Lazuli (Kelompok 4) dari kelas MOLECULE program "AI 4 Jobs"')
    
    team_members = [
        {
            'name': 'Maharani Dwisetia Sri Rezeki',
            'nickname': 'Rani',
            'asal kampus': 'Universitas Bina Nusantara',
            'photo': 'rani.jpg',
            'instagram': 'https://instagram.com/maharani.rezeki',
        },
        {
            'name': 'Atthilla Sulthan Ramadhan',
            'nickname': 'Atthilla',
            'asal kampus': 'Universitas Muhammadiyah Jakarta',
            'photo': 'Cok.jpg',
            'instagram': 'https://instagram.com/atthilla',
        },
        {
            'name': 'Yehezkiel Andreas Makarawung',
            'nickname': 'Kiel',
            'asal kampus': 'Universitas Bina Sarana Informatika',
            'photo': 'kiel.jpg',
            'instagram': 'https://instagram.com/yhzkielam',
        },
        {
            'name': 'Firgi Saridaningsih',
            'nickname': 'Igi',
            'asal kampus': 'Universitas Pendidikan Indonesia',
            'photo': 'igi.jpg',
            'instagram': 'https://instagram.com/firgisarida',
        },
        {
            'name': 'Nisrina Qonita',
            'nickname': 'Nisrina',
            'asal kampus': 'Institut Bisnis & Informatika Kosgoro',
            'photo': 'nisrina.jpg',
            'instagram': 'https://instagram.com/nisrinataa_',
        }
    ]

    for team_member in team_members:
        col1, col2 = st.columns(2)
        with col1:
            st.image(team_member['photo'], width=150)
        with col2:
            st.subheader(team_member['name'])
            st.write(f"Asal kampus: {team_member['asal kampus']}")
            st.write(f"Instagram: [{team_member['nickname']}'s Instagram]({team_member['instagram']})")
        st.markdown("<style>img{border-radius: 10px; box-shadow: 5px 5px 5px #888888;}</style>", unsafe_allow_html=True)
