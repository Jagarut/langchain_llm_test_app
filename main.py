import langchain_helper as lch
import streamlit as st

# Define color options for each animal type
color_options = {
    "Cat": ("White", "Black", "Brown", "Yellow"),
    "Dog": ("White", "Black", "Brown", "Yellow"),
    "Cow": ("White", "Black", "Brown", "Yellow"),
    "Sheep": ("White", "Black", "Brown", "Yellow"),
}

st.title("Pet Name Generator")

animal_type = st.sidebar.selectbox("What is your pet?", ("Cat", "Dog", "Cow", "Sheep")) 

# Get the color options based on the animal type
if animal_type in color_options:
    pet_color = st.sidebar.selectbox(
        f"What color is your {animal_type.lower()}?", color_options[animal_type]
    )

if st.sidebar.button("🚀 Generate names"):
    st.spinner("Generating names...")
    response = lch.generate_pet_name(animal_type, pet_color)
    st.text(response)
