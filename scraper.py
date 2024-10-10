# scrape.py

from bs4 import BeautifulSoup
import requests

def scrape_doctor_data(location, specialization):
    """
    Scrapes doctor data from Practo based on the given location and specialization.
    """
    print(f'Searching for doctors in {location} with specialization: {specialization}')

    # Fetching URL
    # Replace spaces with '%20' for URL encoding
    location_encoded = location.replace(' ', '%20')
    specialization_encoded = specialization.replace(' ', '%20')
    
    url = f'https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22{specialization_encoded}%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city={location_encoded}'
    
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f'Failed to fetch data from Practo, status code: {response.status_code}')
        return []

    # Scraping begins
    soup = BeautifulSoup(response.text, 'html.parser')
    info = soup.find_all('div', class_="info-section")

    # List to hold all doctors' data
    doctor_data = []

    for i in info:
        doctor_info = {}

        # Extracting doctor's name
        name = i.find('h2', class_="doctor-name")
        doctor_info['Name'] = name.text.strip() if name else ""

        # Extracting link to doctor's profile
        link = i.find('a', href=True)
        doctor_info['About Doctor'] = f"https://www.practo.com{link['href']}" if link and link['href'] else ""

        # Extracting specialization
        desgn = i.find('span')
        doctor_info['Specialization'] = desgn.text.strip() if desgn else ""

        # Extracting experience
        exp = i.find('div', class_="uv2-spacer--xs-top")
        experience_text = exp.text.strip().replace('\xa0', ' ') if exp else ""
        doctor_info['Experience'] = experience_text

        # Extracting location
        loc = i.find('div', class_="u-bold u-d-inlineblock u-valign--middle")
        location_parts = loc.find_all('span') if loc else []
        location = ' '.join(part.text.strip() for part in location_parts) if loc else ""
        doctor_info['Location'] = location

        # Extracting consultation fee
        fee = i.find('span', {'data-qa-id': 'consultation_fee'})
        doctor_info['Consultation Fee'] = fee.text.strip() if fee else ""

        # Append doctor info to the list only if all required fields are present
        if all(value != "" for value in doctor_info.values()):
            doctor_data.append(doctor_info)

    return doctor_data

# Example usage (uncomment to test):
# print(scrape_doctor_data("Bangalore", "dentist"))
