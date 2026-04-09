from googletrans import Translator
import glob
import asyncio

lang = 'spanish'
fnames = glob.glob(f'data/*/*{lang}*.txt', recursive=True)

txt = ''

for file in fnames:
    with open(file, 'rb') as f:
        txt += f.read().decode().strip()

txt = txt.replace("\r", "").replace('\n', '')
txt
sample = txt[:500]

async def translate_text(sample, src):
    # print('>', sample)
    if src == 'en':
        return sample
    async with Translator() as translator:
        result = await translator.translate(sample, src=src, dest='en')
        if result.text == sample:
            raise Exception('Translation failed. Try lowering the buffer with "--buffer 1000".')    
        
    return result.text
        
if __name__ == '__main__':
    translated = asyncio.run(translate_text(sample, 'zh-cn'))
    print(translated)