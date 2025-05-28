import os
import random
from tqdm import tqdm
from dataio import read_file
from dataio import write_file

# List of 50 sentence templates
PROMPT_TEMPLATE_LIST = [
    '玉山銀行定期舉辦法人說明會，向法人及投資者報告最新財務業績和未來發展計畫。此活動提供即時的市場資訊，有助於增進雙方的溝通與合作。有出現: {}. 開始吧.',

    '在玉山銀行的法人說明會上，銀行管理層會分享季度財報及最新的營運策略。這讓投資者能掌握企業動向，做出更明智的投資決策。有出現: {}. 開始吧.',

    '玉山銀行舉辦的法人說明會，是銀行向法人客戶及股東闡述經營現況及未來規劃的重要平台。通過會議，法人客戶可以了解公司的長期發展藍圖。有出現: {}. 開始吧.',

    '法人說明會是玉山銀行與法人客戶直接溝通的重要管道，提供最新的財務報告和經營策略。透過詳細的數據分析，銀行展現對市場的洞察力。有出現: {}. 開始吧.',

    '玉山銀行的法人說明會致力於向法人說明公司的財務狀況、經營策略及市場預期。這是銀行向市場展示透明度和穩定成長的機會。有出現: {}. 開始吧.',

    '每季的玉山銀行法人說明會是法人了解銀行經營狀況的重要渠道。會中討論未來發展策略，為投資者提供決策依據。有出現: {}. 開始吧.',

    '玉山銀行在法人說明會上介紹了最新的營運績效和發展計畫，強調對市場趨勢的掌握。這讓法人客戶能及時調整他們的投資策略。有出現: {}. 開始吧.',

    '玉山銀行定期舉辦法人說明會，讓法人投資者了解最新的財務狀況和業務發展。這場會議也提供了雙向交流的機會，有助於強化投資者信心。有出現: {}. 開始吧.',

    '法人說明會是玉山銀行對外展示業績與策略的重要活動，涵蓋最新的財務表現及市場分析。透過說明會，法人客戶得以掌握公司的核心價值。有出現: {}. 開始吧.',

    '玉山銀行法人說明會深入解讀財報，並分享對經濟環境的見解。此舉有助於建立銀行在法人心中的信任和品牌形象。有出現: {}. 開始吧.',

    '玉山銀行的法人說明會提供了詳盡的財務數據和業務策略，確保法人投資者獲得最新資訊。透過這個平台，銀行與法人建立起更緊密的合作關係。有出現: {}. 開始吧.',

    '在法人說明會上，玉山銀行管理層會詳細介紹公司當前的業務狀況及未來發展計畫。這有助於法人理解銀行的經營策略及市場定位。有出現: {}. 開始吧.',

    '玉山銀行舉辦的法人說明會旨在加強與法人投資者的溝通，提供全面的業務及財務狀況報告。這也是銀行展示透明經營的最佳時機。有出現: {}. 開始吧.',

    '每次的法人說明會，玉山銀行都會針對市場趨勢及經營策略做全面分析，提升法人客戶的投資信心。這是投資者了解公司現況的有效途徑。有出現: {}. 開始吧.',

    '玉山銀行透過法人說明會，與法人客戶分享最新的營運成績及策略調整方向。這一平台能夠有效傳達公司的長期經營願景。有出現: {}. 開始吧.',

    '法人說明會是玉山銀行向法人揭示經營現狀及策略佈局的關鍵活動。透過詳細的數據和分析，投資者能夠更深入了解銀行的競爭優勢。有出現: {}. 開始吧.',

    '玉山銀行的法人說明會提供法人投資者全面的財務報告及未來策略，增強市場信心。這是一個雙方交流、探討合作機會的重要場合。有出現: {}. 開始吧.',

    '在玉山銀行的法人說明會中，銀行會公開最近的財務表現，並解釋未來的營運計劃。這為法人提供了可靠的參考，幫助他們作出投資決策。有出現: {}. 開始吧.',

    '法人說明會讓玉山銀行可以向法人客戶解釋公司的財務策略及市場佈局。這是一個展示企業實力和市場遠見的重要場合。有出現: {}. 開始吧.',

    '玉山銀行每次的法人說明會，均致力於提供法人客戶最新的財務狀況及策略調整方向。這樣的資訊分享對於強化投資者信心極為重要。有出現: {}. 開始吧.',

    '玉山銀行在法人說明會上詳細介紹公司經營狀況、財務績效和未來發展計畫。這為法人投資者提供一個全面了解公司的窗口。有出現: {}. 開始吧.',

    '法人說明會是玉山銀行與法人客戶深入交流的平台，展示最新的財務報告及營運策略。銀行藉此活動展現了其市場前瞻性和經營決心。有出現: {}. 開始吧.',

    '玉山銀行的法人說明會，旨在向法人及投資者報告公司的營運狀況和未來展望。這一平台能讓投資者深入了解銀行的發展策略。有出現: {}. 開始吧.',

    '在法人說明會中，玉山銀行會分享最新的市場動態及財務數據，增強法人客戶對企業的信任。這也是銀行對外展示透明經營的重要方式。有出現: {}. 開始吧.',

    '玉山銀行舉辦的法人說明會，專注於向法人投資者報告最新的財務狀況和業務發展。這讓法人能更好地了解公司的經營策略及未來展望。有出現: {}. 開始吧.',

    '法人說明會是玉山銀行定期舉辦的重要活動，提供法人客戶最新的財務狀況及策略分析。此活動促進了銀行與投資者之間的信任與合作。有出現: {}. 開始吧.',

    '在玉山銀行的法人說明會上，會詳細討論公司當前的經營策略及未來規劃。這讓法人投資者能夠掌握公司的動向，制定最佳的投資計劃。有出現: {}. 開始吧.',

    '玉山銀行透過法人說明會，與法人分享最新的市場趨勢及財務分析。這不僅增強投資者的信心，也有助於銀行拓展市場。有出現: {}. 開始吧.',

    '每季舉辦的法人說明會是玉山銀行對外發表經營成果的關鍵時刻。透過這樣的公開交流，法人能夠更深入了解銀行的業務走向。有出現: {}. 開始吧.',

    '玉山銀行的法人說明會提供了與法人客戶直接溝通的管道，闡述最新的財務報表及營運方向。透過資訊透明化，促進雙方的合作關係。有出現: {}. 開始吧.',

    '玉山銀行法人說明會的目標是與法人客戶建立透明、公開的溝通機制。這讓投資者能夠瞭解公司的核心價值及長期發展方向。有出現: {}. 開始吧.',

    '在法人說明會中，玉山銀行管理層會介紹最近的經營績效及市場策略。這有助於法人客戶對公司的發展前景有更深入的認識。有出現: {}. 開始吧.',

    '玉山銀行的法人說明會著重於提供最新的財務數據和市場策略，以增強法人投資者的信心。這是一個探討未來合作機會的理想平台。有出現: {}. 開始吧.',

    '每次法人說明會，玉山銀行都會針對最新的經營狀況和財務數據進行報告。此舉有助於投資者了解公司的長期發展計劃。有出現: {}. 開始吧.',

    '玉山銀行透過法人說明會，向法人投資者展示公司的經營策略和市場佈局。這種定期報告的方式，有助於維持投資者的信任感。有出現: {}. 開始吧.',

    '法人說明會是玉山銀行定期向法人客戶報告經營狀況的重要活動，涵蓋最新的市場資訊。這也為雙方提供了一個交流合作的契機。有出現: {}. 開始吧.',

    '在法人說明會上，玉山銀行詳細闡述公司的發展策略和市場動態。這為法人客戶提供了充分的資訊，助其做出投資決策。有出現: {}. 開始吧.',

    '玉山銀行定期舉辦法人說明會，讓法人投資者掌握公司的最新財務狀況及經營策略。這也是銀行展示未來願景的重要機會。有出現: {}. 開始吧.',

    '法人說明會是玉山銀行向法人投資者報告經營成果及未來規劃的管道。這有助於強化法人對公司的認識，並增進合作信心。有出現: {}. 開始吧.',

    '玉山銀行的法人說明會為法人客戶提供深入了解公司經營狀況的機會。這場會議也展示了銀行對市場的深刻洞察及策略佈局。有出現: {}. 開始吧.',

    '玉山銀行透過法人說明會，向法人客戶公開財務報表及策略方向，提升雙方互信。這有助於投資者評估銀行的發展潛力。有出現: {}. 開始吧.',

    '法人說明會是玉山銀行與法人客戶直接對話的重要平台，分享最新的財務數據及市場策略。銀行藉此機會加強與投資者的合作關係。有出現: {}. 開始吧.',

    '玉山銀行舉辦的法人說明會，提供最新的營運資訊及市場趨勢，增強投資者對企業的信任。這是公司展示經營成果的重要管道。有出現: {}. 開始吧.',

    '在法人說明會上，玉山銀行管理層會詳細說明經營策略及未來發展計畫。這讓法人投資者能對公司的長期規劃有更深入的了解。有出現: {}. 開始吧.',

    '玉山銀行的法人說明會讓法人客戶掌握公司的經營狀況及財務成果。這也是銀行向外界傳達透明化經營的重要方式。有出現: {}. 開始吧.',

    '玉山銀行定期舉辦法人說明會，提供法人投資者最新的財務狀況及策略方向。這為雙方的長期合作奠定了堅實基礎。有出現: {}. 開始吧.',

    '法人說明會是玉山銀行與法人投資者分享最新經營策略及市場預測的關鍵活動。透過深入交流，雙方得以共同探索未來合作機會。有出現: {}. 開始吧.',

    '在法人說明會中，玉山銀行會詳細介紹最新的市場策略及營運績效。這讓法人客戶更全面地了解公司的發展方向。有出現: {}. 開始吧.',

    '玉山銀行的法人說明會為法人投資者提供最新的經營狀況及市場趨勢，提升投資決策的準確性。這是雙方交流的重要平台。有出現: {}. 開始吧.',

    '法人說明會是玉山銀行與法人投資者分享財務狀況及策略方向的場合。透過這樣的公開報告，銀行展示了其在市場上的競爭力。有出現: {}. 開始吧.',
]
# List of 5 unrelated sentences
UNRELATED_SENTENCES = [
    '今天我們要介紹的是花蓮的太魯閣國家公園，這裡以峽谷和瀑布景觀聞名，是登山和健行愛好者的天堂。有出現: {}. 開始吧.',
    '提起義大利美食，披薩和義大利麵是絕對不可錯過的經典菜餚。有出現: {}. 開始吧.',
    '獅子座的人通常充滿自信，擁有領導能力，喜歡成為人群中的焦點。有出現: {}. 開始吧.',
    '在繪畫中，水彩畫以其透明和清新見長，能表現出自然景物的柔和與細膩。有出現: {}. 開始吧.',
    '玉山主峰是台灣最高的山峰，海拔3,952公尺，氣勢磅礡，是登山愛好者挑戰自我的絕佳選擇。有出現: {}. 開始吧.'
]
TRAIN_DEV_BLIST_PATH = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/rareword_f1000_train.txt"
TEST_BLIST_PATH      = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/rareword_f10_test.txt"

# PROMPT_TEMPLATE         = '''玉山銀行法人說明會是一個定期舉辦的活動，向投資者和法人機構介紹公司財務狀況、業務策略及未來展望。該活動旨在增進透明度，讓法人了解公司的經營績效與市場動態。有出現: {}. 開始吧.'''
PROMPT_NON_ENT_TEMPLATE = '''開始吧.'''

def get_uttblist(words, blist):
    return [[str(word2idx[word]), word] for word in words if word in blist]
def generate_prompt(uttblist, blist, shuffle_blist=False):
    # 隨機補充uttblist至10個不重複的詞
    if len(uttblist) < 10:
        additional_words = random.sample([word for word in blist if word not in uttblist], 10 - len(uttblist))
        uttblist.extend(additional_words)
    
    # Shuffle the uttblist if shuffle_blist is True
    if shuffle_blist and len(uttblist) > 0:
        random.shuffle(uttblist)

    # 5% chance to select an unrelated sentence
    if random.random() < 0.05:
        unrelated_template = random.choice(UNRELATED_SENTENCES)
        return unrelated_template.format(", ".join(uttblist))
    
    # Otherwise, select from the regular templates
    if len(uttblist) > 0:
        selected_template = random.choice(PROMPT_TEMPLATE_LIST)
        return selected_template.format(", ".join(uttblist))
    else:
        return PROMPT_NON_ENT_TEMPLATE


if __name__ == '__main__':
    datas_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/dump/raw'
    for folder in os.listdir(datas_path):
        path = os.path.join(datas_path, folder)
        if not os.path.isfile(os.path.join(path, 'wav.scp')):
            continue
        if 'test' not in path:
            blist_path = TRAIN_DEV_BLIST_PATH
        else:
            blist_path = TEST_BLIST_PATH
        blist = [b[0] for b in read_file(blist_path, sp=' ')]
        word2idx = {word: i for i, word in enumerate(blist)}

        print(f'processing {path}...')
        text_path  = os.path.join(path, 'text')
        text_datas = read_file(text_path, sp=' ')
        
        rareword_datas = []
        for data in tqdm(text_datas):
            uttid    = data[0]
            results  = get_uttblist(data[1:], blist)
            uttblist = [d[1] for d in results]
            
            if len(uttblist) > 0:
                # Pass shuffle_blist=True to enable random sorting
                prompt = generate_prompt(uttblist, blist, shuffle_blist=False)

            else:
                prompt = PROMPT_NON_ENT_TEMPLATE
            
            # Append the generated prompt to rareword_datas
            rareword_datas.append([uttid, prompt.upper()])

        # Ensure output path exists before writing
        output_path_uttblist = os.path.join(path, 'prompt_multides_unrelated_random10')
        os.makedirs(os.path.dirname(output_path_uttblist), exist_ok=True)
        
        # Write prompts to file
        write_file(output_path_uttblist, rareword_datas)
