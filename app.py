# import streamlit as st
# from text_to_image import recommend_images_based_on_text
# from image_to_image import recommend_images_based_on_image
# from PIL import Image
# import os

# st.title('Système de Recommandation d\'Images')

# option = st.selectbox(
#     'Choisissez une méthode de recommandation',
#     ('Texte à Image', 'Image à Image')
# )

# if option == 'Texte à Image':
#     st.header('Recommandation d\'Images Basée sur le Texte')

#     prompt = st.text_input('Entrez une description de l\'image')

#     if st.button('Recommander'):
#         if prompt:
#             with st.spinner('Calcul en cours...'):
#                 df_final = recommend_images_based_on_text(prompt)
#             st.write('Images Recommandées:')
#             for i, row in df_final.iterrows():
#                 st.image(row["Image Path"], caption=row["Caption"], use_column_width=True)
#         else:
#             st.error("Veuillez entrer une description.")

# elif option == 'Image à Image':
#     st.header('Recommandation d\'Images Basée sur une Image')

#     uploaded_file = st.file_uploader("Choisissez une image de référence", type=['jpg', 'png'])
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Image Téléchargée', use_column_width=True)

#         if st.button('Recommander'):
#             reference_image_path = os.path.join(os.getcwd(), uploaded_file.name)
#             with open(reference_image_path, 'wb') as f:
#                 f.write(uploaded_file.getbuffer())

#             with st.spinner('Calcul en cours...'):
#                 similar_images = recommend_images_based_on_image(reference_image_path)
#             st.write('Images Recommandées:')
#             for img_path in similar_images:
#                 st.image(img_path, use_column_width=True)

# Pour exécuter l'application, utilisez la commande suivante dans le terminal:
# streamlit run app.py
import streamlit as st
from text_to_image import recommend_images_based_on_text
from image_to_image import get_image_paths, recommend_images_based_on_image, cluster_images, visualize_clusters
from PIL import Image
import os

st.title('Système de Recommandation d\'Images')

option = st.selectbox(
    'Choisissez une méthode de recommandation',
    ('Texte à Image', 'Image à Image')
)

categories = {
    "shirt": "Ao",
    "backpack": "Balo",
    "water animals": "Ca",
    "wands": "CanCau",
    "dogs": "Cho",
    "chairs": "Ghe",
    "shoes": "Giay",
    "beds": "Giuong",
    "pigs": "Heo",
    "divers": "Khac",
    "cubes": "Khoi",
    "glasses": "Kinh",
    "triangles": "Leu",
    "cats": "Meo",
    "hats": "Non",
    "pants": "Quan",
    "bunnies": "Tho",
    "hair cuts": "Toc",
    "eggs": "Trung",
    "skirts": "Vay",
    "cars": "Xe"
}

if option == 'Texte à Image':
    st.header('Recommandation d\'Images Basée sur le Texte')

    category = st.selectbox('Choisissez une catégorie', list(categories.keys()))
    
    if category:
        prompt = st.text_input('Entrez une description de l\'image')

        if st.button('Recommander'):
            if prompt:
                with st.spinner('Calcul en cours...'):
                    df_final = recommend_images_based_on_text(prompt, categories[category])
                st.write('Images Recommandées:')
                for i, row in df_final.iterrows():
                    st.image(row["Image Path"], caption=row["Caption"], use_column_width=True)
            else:
                st.error("Veuillez entrer une description.")

elif option == 'Image à Image':
    st.header('Recommandation d\'Images Basée sur une Image')

    category = st.selectbox('Choisissez une catégorie', list(categories.keys()))
    
    if category:
        image_dir = './train_cleaned'
        image_paths = get_image_paths(image_dir, categories[category])
        
        if image_paths:
            st.write(f"Images dans la catégorie '{category}':")
            
            selected_image_path = st.selectbox(
                'Choisissez une image de référence',
                image_paths
            )
            
            if selected_image_path:
                image = Image.open(selected_image_path)
                st.image(image, caption='Image Sélectionnée', use_column_width=True)
            
                if st.button('Recommander'):
                    with st.spinner('Calcul en cours...'):
                        similar_images = recommend_images_based_on_image(selected_image_path, image_dir, categories[category])
                    st.write('Images Recommandées:')
                    for img_path in similar_images:
                        st.image(img_path, use_column_width=True)
        else:
            st.write("Aucune image trouvée dans cette catégorie.")

    if st.button('Visualiser les Clusters'):
        if category:
            image_paths = get_image_paths(image_dir, categories[category])
            if image_paths:
                clusters = cluster_images(image_paths)
                st.write('Visualisation des Clusters:')
                visualize_clusters(image_paths, clusters)
            else:
                st.write("Aucune image trouvée dans cette catégorie.")

# Pour exécuter l'application, utilisez la commande suivante dans le terminal:
# streamlit run app.py
