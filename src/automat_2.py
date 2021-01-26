from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time

usuario="115646575"
passw="Belug@07"

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get("https://accounts.claveunica.gob.cl/accounts/login/?next=/openid/authorize%3Fclient_id%3D347178b052294153af290d6a84589868%26redirect_uri%3Dhttps%253A%252F%252Fagenda.minpublico.cl%252Fapp%252Ffiscalia-en-linea%252Fresp-cu%26response_type%3Dcode%26scope%3Dopenid%2520run%2520name%26state%3D0KglgOwXrNkbKJPpLitqE9avpb964j4JqY0P6Z2lhGBoTHQMzplEWGrCYDJCFEus")

mBox = driver.find_element_by_xpath("/html/body/main/section/div/div[1]/div/form/div[1]/input[1]")
mBox.send_keys(usuario)
mBox = driver.find_element_by_xpath("/html/body/main/section/div/div[1]/div/form/div[2]/input")
mBox.send_keys(passw)

mBox = driver.find_element_by_xpath("/html/body/main/section/div/div[1]/div/form/div[3]").click()

mBox = driver.find_element_by_xpath("/html/body/div[2]/div/button[1]").click() #2nd click


