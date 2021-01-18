from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time

usuario="115646575"
passw="Belug@07"

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get("https://accounts.claveunica.gob.cl/accounts/login/?next=/openid/authorize%3Fclient_id%3Dd602a0071f3f4db8b37a87cffd89bf23%26redirect_uri%3Dhttps%253A%252F%252Foficinajudicialvirtual.pjud.cl%252Fclaveunica%252Freturn.php%26response_type%3Dcode%26scope%3Dopenid%2Brut%26state%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwczpcL1wvb2ZpY2luYWp1ZGljaWFsdmlydHVhbC5wanVkLmNsIiwiYXVkIjoiaHR0cHM6XC9cL29maWNpbmFqdWRpY2lhbHZpcnR1YWwucGp1ZC5jbCIsImlhdCI6MTYxMDk4MDE4NiwiZXhwIjoxNjEwOTgxMDg2LCJkYXRhIjp7InNlc3Npb25pZCI6MzExOTU1NTF9fQ.xPn3-c21UOoxyKB3JmhV3fjuMClDS1kWeMlwhbXPGP0")
time.sleep(2)

mBox = driver.find_element_by_xpath("/html/body/main/section/div/div[1]/div/form/div[1]/input[1]")
mBox.send_keys(usuario)
mBox = driver.find_element_by_xpath("/html/body/main/section/div/div[1]/div/form/div[2]/input")
mBox.send_keys(passw)

mBox = driver.find_element_by_xpath("/html/body/main/section/div/div[1]/div/form/div[3]/div").click()


