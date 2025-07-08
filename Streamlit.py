import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 🧾 Názvy tříd podle modelu
class_names = class_names = ['Abra',
 'Aerodactyl',
 'Alakazam',
 'Arbok',
 'Arcanine',
 'Articuno',
 'Beedrill',
 'Bellsprout',
 'Blastoise',
 'Bulbasaur',
 'Butterfree',
 'Caterpie',
 'Chansey',
 'Charizard',
 'Charmander',
 'Charmeleon',
 'Clefable',
 'Clefairy',
 'Cloyster',
 'Cubone',
 'Dewgong',
 'Diglett',
 'Ditto',
 'Dodrio',
 'Doduo',
 'Dragonair',
 'Dragonite',
 'Dratini',
 'Drowzee',
 'Dugtrio',
 'Eevee',
 'Ekans',
 'Electabuzz',
 'Electrode',
 'Exeggcute',
 'Exeggutor',
 'Farfetch',
 'Fearow',
 'Flareon',
 'Gastly',
 'Gengar',
 'Geodude',
 'Gloom',
 'Golbat',
 'Goldeen',
 'Golduck',
 'Golem',
 'Graveler',
 'Grimer',
 'Growlithe',
 'Gyarados',
 'Haunter',
 'Hitmonchan',
 'Hitmonlee',
 'Horsea',
 'Hypno',
 'Ivysaur',
 'Jigglypuff',
 'Jolteon',
 'Jynx',
 'Kabuto',
 'Kabutops',
 'Kadabra',
 'Kakuna',
 'Kangaskhan',
 'Kingler',
 'Koffing',
 'Krabby',
 'Lapras',
 'Lickitung',
 'Machamp',
 'Machoke',
 'Machop',
 'Magikarp',
 'Magmar',
 'Magnemite',
 'Magneton',
 'Mankey',
 'Marowak',
 'Meowth',
 'Metapod',
 'Mew',
 'Mewtwo',
 'Moltres',
 'Mr-Mime',
 'Muk',
 'Nidoking',
 'Nidoqueen',
 'Nidoran-f',
 'Nidoran-m',
 'Nidorina',
 'Nidorino',
 'Ninetales',
 'Oddish',
 'Omanyte',
 'Omastar',
 'Onix',
 'Paras',
 'Parasect',
 'Persian',
 'Pidgeot',
 'Pidgeotto',
 'Pidgey',
 'Pikachu',
 'Pinsir',
 'Poliwag',
 'Poliwhirl',
 'Poliwrath',
 'Ponyta',
 'Porygon',
 'Primeape',
 'Psyduck',
 'Raichu',
 'Rapidash',
 'Raticate',
 'Rattata',
 'Rhydon',
 'Rhyhorn',
 'Sandshrew',
 'Sandslash',
 'Scyther',
 'Seadra',
 'Seaking',
 'Seel',
 'Shellder',
 'Slowbro',
 'Slowpoke',
 'Snorlax',
 'Spearow',
 'Squirtle',
 'Starmie',
 'Staryu',
 'Tangela',
 'Tauros',
 'Tentacool',
 'Tentacruel',
 'Vaporeon',
 'Venomoth',
 'Venonat',
 'Venusaur',
 'Victreebel',
 'Vileplume',
 'Voltorb',
 'Vulpix',
 'Wartortle',
 'Weedle',
 'Weepinbell',
 'Weezing',
 'Wigglytuff',
 'Zapdos',
 'Zubat']  # uprav dle svého modelu

# 🧠 Načti model
model = tf.keras.models.load_model("soubor_densenet121.keras")  # uprav podle skutečné cesty

# 🔧 Funkce pro zpracování obrázku
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# 🖼️ GUI
st.title("🧠 Detekce Pokémona pomocí neuronové sítě")
st.markdown("Nahraj obrázek a zjisti, **který Pokémon** se na něm skrývá!")

uploaded_file = st.file_uploader("📁 Vyber obrázek", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="📸 Nahraný obrázek", use_container_width=True)

    # Ulož si vstupní data pro další použití
    st.session_state.input_data = preprocess_image(image)

    if st.button("🔍 Spustit predikci"):
        prediction = model.predict(st.session_state.input_data)
        predicted_index = np.argmax(prediction)
        predicted_name = class_names[predicted_index]
        confidence = prediction[0][predicted_index]

        st.success(f"🎯 Detekovaný Pokémon: **{predicted_name}**")
        st.write(f"📊 Pravděpodobnost: `{confidence * 100:.2f}%`")

        # Pro ladění nebo zobrazení celého výstupu:
        # st.write("🔬 Kompletní predikce:", prediction)