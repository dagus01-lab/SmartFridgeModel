import csv
import os
import random
from io import BytesIO

import requests
import time

from PIL import Image
from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.parse import urljoin


# Function to download images from a webpage
def download_images_esselunga(url, save_folder):
    # Create a directory to save images if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Initialize the Chrome webdriver
    driver = webdriver.Chrome()

    # Open the webpage
    driver.get(url)

    # Wait for the page to fully load (you might need to adjust the time depending on the page)
    time.sleep(10)

    # Get the page source after it has fully loaded
    page_source = driver.page_source

    # Close the webdriver
    driver.quit()

    # Parse the HTML content of the page
    soup = BeautifulSoup(page_source, 'html.parser')

    product_img_containers = soup.find_all('a', {'class': 'el-link'})

    # Download images within each product-img-container
    for container in product_img_containers:
        download_images_from_container(container, url, save_folder)


def download_images_coop(url, save_folder, product_name, base_url="https://www.coopshop.it", max=15):
    # Create a directory to save images if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Initialize the Chrome webdriver
    driver = webdriver.Chrome()

    # Open the webpage
    driver.get(url)

    # Wait for the page to fully load (you might need to adjust the time depending on the page)
    time.sleep(5)

    # Get the page source after it has fully loaded
    page_source = driver.page_source

    # Close the webdriver
    driver.quit()

    # Parse the HTML content of the page
    soup = BeautifulSoup(page_source, 'html.parser')
    product_table = soup.find_all('div', {'class': 'row'})
    count = 0
    for product_row in product_table:
        if count >=max:
            break
        prod_pics = product_row.find_all('div', {'class':'w-100'})
        for image_container in prod_pics:
            download_images_from_container(image_container, base_url, save_folder, product_name)

        """
        additional_pics_link = product_row.find_all('a')
        # Iterate over each link to visit it and download images
        for link_element in additional_pics_link:
            if link_element is not None:
                link_url = link_element.get("href")
                if "product" in link_url:
                    download_images_coop_detail_page(base_url+link_url, save_folder, product_name, base_url)"""
        count+=1


def download_images_duckduckgo(url, save_folder):
    # Create a directory to save images if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Initialize the Chrome webdriver
    driver = webdriver.Chrome()

    # Open the webpage
    driver.get(url)

    # Wait for the page to fully load (you might need to adjust the time depending on the page)
    time.sleep(3)

    # Get the page source after it has fully loaded
    page_source = driver.page_source

    # Close the webdriver
    driver.quit()

    # Parse the HTML content of the page
    soup = BeautifulSoup(page_source, 'html.parser')

    product_img_containers = soup.find_all('div', {'class': 'tile-wrap'})

    # Download images within each product-img-container
    for container in product_img_containers:
        download_images_from_container(container, url, save_folder=save_folder, max=20)

def download_images_from_container(container, base_url, save_folder,product_name=None,max=50):
    # Find all image tags within the container
    img_tags = container.find_all('img')
    count = 0

    # Download each image found
    for img_tag in img_tags:
        if count>max:
            break
        alt_tag_name = img_tag.get('alt')
        if alt_tag_name is not None:
            try:
                alt_tag_name = (str(alt_tag_name)).lower()
            except Exception as e:
                alt_tag_name = ""
        else:
            alt_tag_name = ""

        if product_name is None or product_name in alt_tag_name:
            # Get the source URL of the image
            img_url = img_tag.get('src')
            if img_url is None or ( not str(img_url).endswith(".jpg") and not str(img_url).endswith(".png") and
                                    not str(img_url).endswith(".jpeg") and not str(img_url).endswith("images")):
                continue

            # If the URL is relative, convert it to absolute
            img_url = urljoin(base_url, img_url)


            # Get the image filename
            img_filename = os.path.basename(img_url)
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
                'Mozilla/5.0 (iPhone; CPU iPhone OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
                'Mozilla/5.0 (Linux; Android 11; SM-G960U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.72 Mobile Safari/537.36'
            ]
            user_agent = random.choice(user_agents)
            headers = {'User-Agent': user_agent}

            # Download the image
            img_data = requests.get(img_url, headers=headers).content
            image = Image.open(BytesIO(img_data))
            if is_white_background(image):
                with open(os.path.join(save_folder, img_filename), 'wb') as img_file:
                    img_file.write(img_data)
                print(f"Downloaded: {img_filename}")
                count+=1
                time.sleep(1)

def read_products_file(file_path):
    data = []
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            # Assuming the CSV file has two columns: product and class name
            product = row[0]
            class_name = row[1]
            subqueries = row[2:]
            data.append((product, class_name, subqueries))
    return data


def is_white_background(image, border_size=5, tolerance=35):
    """
    Check if the background of the image is white.

    Args:
    - image: PIL Image object.
    - border_size: Size of the border (in pixels) to check for white color.
    - tolerance: Tolerance for considering a color as white (0 to 255).

    Returns:
    - True if the background is white, False otherwise.
    """
    # Convert the image to grayscale for easier comparison
    grayscale_image = image.convert('L')

    # Get the size of the image
    width, height = grayscale_image.size

    # Define the region to check (border area)
    border_region = (0,0,width-1, height-1)

    # Get the colors of the border region
    border_colors = grayscale_image.crop(border_region).getcolors()

    # Check if all border colors are within the tolerance of white
    for count, color in border_colors:
        if color < tolerance:
            return False

    return True

if __name__ == "__main__":
    # URL of the website containing the food catalogue
    base_url_esselunga = "https://spesaonline.esselunga.it/commerce/nav/auth/supermercato/store.html?freevisit=true&#!/negozio/ricerca/PRODUCT?facet=1"
    base_url_coop = "https://www.coopshop.it/search?q=PRODUCT"
    base_url_duckduckgo = "https://duckduckgo.com/?q=PRODUCT&t=ftsa&iar=images&iax=images&ia=images"

    products_data = read_products_file("products.csv")
    for product, class_name, subqueries in products_data:
        # Download images from the various urls to be scraped
        url_esselunga = base_url_esselunga.replace("PRODUCT", product)
        url_coop = base_url_coop.replace("PRODUCT", product)
        product_name = product.lower().split()[0]
        for subquery in subqueries:
            url_duckduckgo = base_url_duckduckgo.replace("PRODUCT", subquery)
            download_images_duckduckgo(url_duckduckgo, "../dataset/products/"+class_name)
        #download_images_esselunga(url_esselunga, ".."+os.pathsep+"dataset"+os.pathsep+class_name)
        #download_images_coop(url_coop, ".."+os.pathsep+"dataset"+os.pathsep+class_name, product_name)
        time.sleep(2)
