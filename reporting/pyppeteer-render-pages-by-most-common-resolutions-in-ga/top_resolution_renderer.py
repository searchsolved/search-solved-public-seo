####################################################################################
# Website  : https://leefoot.co.uk/                                                #
# Contact  : https://leefoot.co.uk/hire-me/                                        #
# LinkedIn : https://www.linkedin.com/in/lee-foot/                                 #
# Twitter  : https://twitter.com/LeeFootSEO                                        #
####################################################################################

import pandas as pd
import asyncio
from pyppeteer import launch
import os
from glob import glob

url = "http://www.guardian.co.uk"

dir = os.getcwd()
os.chdir(dir)

for f in glob('*Browser & OS*.csv'):
    df_ga = pd.read_csv((f), skiprows=6)

df_ga.drop(df_ga.loc[10:].index, inplace=True)
df_ga['width'] = df_ga["Screen Resolution"].str.split("x").str[0].astype(int)
df_ga['height'] = df_ga["Screen Resolution"].str.split("x").str[1].astype(int)
width_list = list(df_ga['width'])
height_list = list(df_ga['height'])

async def main():
    browser = await launch()
    page = await browser.newPage()
    await page.setViewport({'width': ga_width, 'height': ga_height})

    await page.goto(url)
    await page.screenshot({'path': str(ga_width) + "x" + str(ga_height) + ".png"})
    dimensions = await page.evaluate('''() => {
        return {
            width: document.documentElement.clientWidth,
            height: document.documentElement.clientHeight,
            deviceScaleFactor: window.devicePixelRatio,
        }
    }''')

    print(dimensions)
    await browser.close()


for ga_width,ga_height in zip (width_list, height_list):
    asyncio.get_event_loop().run_until_complete(main())
