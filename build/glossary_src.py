"""E-commerce glossary, hand-written. English term -> forms per language.
Column order: zh es pt ar ru ja ko de fr hi tr id vi it pl th
Inside a cell, alternatives / inflections are separated by '|'. Empty cell = no entry.
Used by the Fast model as a word/phrase dictionary (longest match first).
"""
LANGS = ["zh", "es", "pt", "ar", "ru", "ja", "ko", "de", "fr", "hi", "tr", "id", "vi", "it", "pl", "th"]

G = [
 # ---- people / gender
 ("women's", "女士|女式|女款|女", "mujer|para mujer|de mujer|dama", "feminino|feminina|para mulher|mulher", "نسائي|نسائية|للنساء", "женская|женский|женские|женское|для женщин", "レディース|女性用", "여성용|여성|여자", "damen|frauen", "femme|pour femme|femmes", "महिलाओं के लिए|महिला", "kadın|bayan", "wanita|perempuan", "nữ", "donna|da donna", "damska|damski|damskie|dla kobiet", "ผู้หญิง|สตรี"),
 ("men's", "男士|男式|男款|男", "hombre|para hombre|de hombre|caballero", "masculino|masculina|para homem|homem", "رجالي|رجالية|للرجال", "мужская|мужской|мужские|мужское|для мужчин", "メンズ|男性用", "남성용|남성|남자", "herren|männer", "homme|pour homme|hommes", "पुरुषों के लिए|पुरुष", "erkek|bay", "pria|laki-laki", "nam", "uomo|da uomo", "męska|męski|męskie|dla mężczyzn", "ผู้ชาย|บุรุษ"),
 ("children's", "儿童|童装|孩子", "niño|niños|infantil|para niños", "infantil|criança|crianças", "أطفال|للأطفال", "детская|детский|детские|для детей", "キッズ|子供用|子供", "아동|어린이|키즈", "kinder", "enfant|enfants|pour enfants", "बच्चों के लिए|बच्चे", "çocuk", "anak|anak-anak", "trẻ em", "bambino|bambini|per bambini", "dziecięca|dziecięcy|dziecięce|dla dzieci", "เด็ก"),
 ("baby", "婴儿|宝宝", "bebé|bebe", "bebê|bebé", "رضيع|طفل رضيع", "детское|для малышей|малыш", "ベビー|赤ちゃん", "아기|유아", "baby", "bébé", "शिशु", "bebek", "bayi", "em bé", "neonato", "niemowlęca|niemowlęcy|dla niemowląt", "ทารก"),
 ("unisex", "男女通用", "unisex", "unissex", "للجنسين", "унисекс", "ユニセックス|男女兼用", "남녀공용", "unisex", "unisexe", "यूनिसेक्स", "unisex", "unisex", "unisex", "unisex", "unisex", "ยูนิเซ็กซ์"),
 # ---- clothing
 ("t-shirt", "T恤|t恤", "camiseta", "camiseta", "تي شيرت", "футболка|футболки", "Tシャツ", "티셔츠", "t-shirt|tshirt", "t-shirt|tee-shirt", "टी-शर्ट", "tişört", "kaos", "áo thun|áo phông", "maglietta|t-shirt", "koszulka|t-shirt", "เสื้อยืด"),
 ("shirt", "衬衫", "camisa", "camisa", "قميص", "рубашка|рубашки", "シャツ", "셔츠", "hemd", "chemise", "शर्ट|कमीज़", "gömlek", "kemeja", "áo sơ mi", "camicia", "koszula", "เสื้อเชิ้ต"),
 ("dress", "连衣裙|裙子", "vestido", "vestido", "فستان", "платье|платья", "ワンピース|ドレス", "원피스|드레스", "kleid", "robe", "पोशाक|ड्रेस", "elbise", "gaun|dress", "váy|đầm", "vestito|abito", "sukienka", "ชุดเดรส|เดรส"),
 ("skirt", "半身裙|裙", "falda", "saia", "تنورة", "юбка", "スカート", "치마|스커트", "rock", "jupe", "स्कर्ट", "etek", "rok", "chân váy", "gonna", "spódnica", "กระโปรง"),
 ("trousers", "裤子|长裤", "pantalón|pantalones", "calça|calças", "بنطلون|بنطال", "брюки|штаны", "パンツ|ズボン", "바지", "hose", "pantalon", "पैंट|पतलून", "pantolon", "celana", "quần", "pantaloni", "spodnie", "กางเกง"),
 ("jeans", "牛仔裤", "vaqueros|jeans", "calça jeans|jeans", "جينز", "джинсы", "ジーンズ", "청바지", "jeans", "jean|jeans", "जींस", "kot|kot pantolon", "jeans", "quần jean", "jeans", "dżinsy|jeansy", "กางเกงยีนส์"),
 ("jacket", "夹克|外套", "chaqueta", "jaqueta", "جاكيت|سترة", "куртка", "ジャケット", "재킷|자켓", "jacke", "veste", "जैकेट", "ceket|mont", "jaket", "áo khoác", "giacca", "kurtka", "เสื้อแจ็คเก็ต"),
 ("coat", "大衣", "abrigo", "casaco", "معطف", "пальто", "コート", "코트", "mantel", "manteau", "कोट", "kaban|palto", "mantel", "áo choàng", "cappotto", "płaszcz", "เสื้อโค้ท"),
 ("sweater", "毛衣", "suéter|jersey", "suéter", "سترة صوف|كنزة", "свитер", "セーター", "스웨터", "pullover", "pull", "स्वेटर", "kazak", "sweter", "áo len", "maglione", "sweter", "เสื้อกันหนาว"),
 ("hoodie", "连帽衫|卫衣", "sudadera", "moletom", "هودي", "толстовка|худи", "パーカー", "후드티", "kapuzenpullover|hoodie", "sweat à capuche", "हुडी", "kapüşonlu sweatshirt", "hoodie", "áo hoodie", "felpa", "bluza z kapturem", "เสื้อฮู้ด"),
 ("socks", "袜子", "calcetines", "meias", "جوارب", "носки", "靴下|ソックス", "양말", "socken", "chaussettes", "मोज़े", "çorap", "kaus kaki", "tất|vớ", "calzini", "skarpetki", "ถุงเท้า"),
 ("underwear", "内衣", "ropa interior", "roupa íntima|cueca|calcinha", "ملابس داخلية", "нижнее бельё|белье", "下着", "속옷", "unterwäsche", "sous-vêtements", "अंडरवियर", "iç çamaşırı", "pakaian dalam", "đồ lót", "intimo", "bielizna", "ชุดชั้นใน"),
 # ---- shoes & accessories
 ("shoes", "鞋|鞋子", "zapatos|zapato", "sapatos|sapato", "أحذية|حذاء", "обувь|туфли", "靴|シューズ", "신발", "schuhe", "chaussures", "जूते", "ayakkabı", "sepatu", "giày", "scarpe", "buty", "รองเท้า"),
 ("sneakers", "运动鞋", "zapatillas", "tênis", "حذاء رياضي", "кроссовки", "スニーカー", "운동화|스니커즈", "turnschuhe|sneaker", "baskets", "स्नीकर्स", "spor ayakkabı", "sepatu kets", "giày thể thao", "sneakers", "trampki|sneakersy", "รองเท้าผ้าใบ"),
 ("boots", "靴子", "botas", "botas", "أحذية طويلة|بوت", "ботинки|сапоги", "ブーツ", "부츠", "stiefel", "bottes", "बूट", "bot|çizme", "sepatu bot", "giày bốt", "stivali", "botki|kozaki", "รองเท้าบูท"),
 ("sandals", "凉鞋", "sandalias", "sandálias", "صنادل", "сандалии|босоножки", "サンダル", "샌들", "sandalen", "sandales", "सैंडल", "sandalet", "sandal", "dép xăng đan", "sandali", "sandały", "รองเท้าแตะ"),
 ("bag", "包|包包", "bolso|bolsa", "bolsa", "حقيبة|شنطة", "сумка|сумки", "バッグ|かばん|鞄", "가방", "tasche", "sac", "बैग|थैला", "çanta", "tas", "túi", "borsa", "torba|torebka", "กระเป๋า"),
 ("handbag", "手提包", "bolso de mano", "bolsa de mão", "حقيبة يد", "сумочка", "ハンドバッグ", "핸드백", "handtasche", "sac à main", "हैंडबैग", "el çantası", "tas tangan", "túi xách", "borsetta", "torebka", "กระเป๋าถือ"),
 ("backpack", "背包|双肩包", "mochila", "mochila", "حقيبة ظهر", "рюкзак", "リュック|バックパック", "백팩|배낭", "rucksack", "sac à dos", "बैकपैक", "sırt çantası", "ransel", "ba lô", "zaino", "plecak", "กระเป๋าเป้"),
 ("wallet", "钱包", "cartera|billetera", "carteira", "محفظة", "кошелёк|кошелек", "財布", "지갑", "geldbörse|portemonnaie", "portefeuille", "बटुआ", "cüzdan", "dompet", "ví", "portafoglio", "portfel", "กระเป๋าสตางค์"),
 ("belt", "腰带|皮带", "cinturón", "cinto", "حزام", "ремень", "ベルト", "벨트", "gürtel", "ceinture", "बेल्ट", "kemer", "ikat pinggang", "thắt lưng", "cintura", "pasek", "เข็มขัด"),
 ("hat", "帽子", "sombrero|gorra", "chapéu|boné", "قبعة", "шапка|шляпа|кепка", "帽子", "모자", "hut|mütze", "chapeau|casquette", "टोपी", "şapka", "topi", "mũ|nón", "cappello", "czapka", "หมวก"),
 ("watch", "手表", "reloj", "relógio", "ساعة", "часы", "腕時計|時計", "시계", "uhr|armbanduhr", "montre", "घड़ी", "saat", "jam tangan", "đồng hồ", "orologio", "zegarek", "นาฬิกา"),
 ("sunglasses", "太阳镜|墨镜", "gafas de sol", "óculos de sol", "نظارة شمسية", "солнцезащитные очки", "サングラス", "선글라스", "sonnenbrille", "lunettes de soleil", "धूप का चश्मा", "güneş gözlüğü", "kacamata hitam", "kính râm", "occhiali da sole", "okulary przeciwsłoneczne", "แว่นกันแดด"),
 ("necklace", "项链", "collar", "colar", "قلادة", "ожерелье|колье", "ネックレス", "목걸이", "halskette", "collier", "हार", "kolye", "kalung", "vòng cổ", "collana", "naszyjnik", "สร้อยคอ"),
 ("ring", "戒指", "anillo", "anel", "خاتم", "кольцо", "指輪|リング", "반지", "ring", "bague", "अंगूठी", "yüzük", "cincin", "nhẫn", "anello", "pierścionek", "แหวน"),
 ("earrings", "耳环", "pendientes|aretes", "brincos", "أقراط", "серьги", "イヤリング|ピアス", "귀걸이", "ohrringe", "boucles d'oreilles", "झुमके|बालियाँ", "küpe", "anting", "bông tai", "orecchini", "kolczyki", "ต่างหู"),
 # ---- electronics
 ("phone", "手机|电话", "teléfono|móvil|celular", "telefone|celular", "هاتف|جوال", "телефон|смартфон", "携帯電話|スマートフォン|スマホ", "휴대폰|핸드폰|스마트폰", "handy|telefon|smartphone", "téléphone|portable", "फ़ोन|मोबाइल", "telefon", "ponsel|telepon|hp", "điện thoại", "telefono|cellulare", "telefon", "โทรศัพท์"),
 ("phone case", "手机壳", "funda para móvil|funda", "capa de celular|capinha", "جراب هاتف|غطاء هاتف", "чехол для телефона|чехол", "スマホケース|携帯ケース", "휴대폰 케이스|폰케이스", "handyhülle", "coque de téléphone|coque", "फ़ोन कवर", "telefon kılıfı|kılıf", "casing hp|case hp", "ốp lưng điện thoại|ốp lưng", "custodia per telefono|cover", "etui na telefon|etui", "เคสโทรศัพท์"),
 ("charger", "充电器", "cargador", "carregador", "شاحن", "зарядное устройство|зарядка", "充電器", "충전기", "ladegerät", "chargeur", "चार्जर", "şarj aleti|şarj cihazı", "pengisi daya|charger", "sạc|bộ sạc", "caricabatterie", "ładowarka", "ที่ชาร์จ"),
 ("cable", "数据线|线", "cable", "cabo", "كابل|سلك", "кабель|провод", "ケーブル", "케이블", "kabel", "câble", "केबल", "kablo", "kabel", "cáp|dây cáp", "cavo", "kabel", "สายเคเบิล|สาย"),
 ("headphones", "耳机", "auriculares", "fones de ouvido", "سماعات", "наушники", "ヘッドホン|イヤホン", "헤드폰|이어폰", "kopfhörer", "écouteurs|casque", "हेडफ़ोन", "kulaklık", "headphone|earphone", "tai nghe", "cuffie|auricolari", "słuchawki", "หูฟัง"),
 ("laptop", "笔记本电脑", "portátil|laptop", "notebook|laptop", "حاسوب محمول|لابتوب", "ноутбук", "ノートパソコン", "노트북", "laptop|notebook", "ordinateur portable", "लैपटॉप", "dizüstü bilgisayar", "laptop", "máy tính xách tay", "portatile", "laptop", "แล็ปท็อป|โน้ตบุ๊ก"),
 ("tablet", "平板电脑", "tableta", "tablet", "جهاز لوحي|تابلت", "планшет", "タブレット", "태블릿", "tablet", "tablette", "टैबलेट", "tablet", "tablet", "máy tính bảng", "tablet", "tablet", "แท็บเล็ต"),
 ("battery", "电池", "batería|pila", "bateria|pilha", "بطارية", "аккумулятор|батарея|батарейка", "バッテリー|電池", "배터리", "akku|batterie", "batterie|pile", "बैटरी", "pil|batarya", "baterai", "pin", "batteria", "bateria", "แบตเตอรี่"),
 ("power bank", "充电宝|移动电源", "batería externa", "carregador portátil|power bank", "باور بانك", "внешний аккумулятор|пауэрбанк", "モバイルバッテリー", "보조배터리", "powerbank", "batterie externe", "पावर बैंक", "taşınabilir şarj cihazı|powerbank", "power bank", "sạc dự phòng", "power bank", "powerbank", "พาวเวอร์แบงค์"),
 ("speaker", "音箱|扬声器", "altavoz", "caixa de som", "مكبر صوت|سماعة", "колонка|динамик", "スピーカー", "스피커", "lautsprecher", "enceinte|haut-parleur", "स्पीकर", "hoparlör", "speaker", "loa", "altoparlante|cassa", "głośnik", "ลำโพง"),
 ("camera", "相机", "cámara", "câmera", "كاميرا", "камера|фотоаппарат", "カメラ", "카메라", "kamera", "appareil photo|caméra", "कैमरा", "kamera|fotoğraf makinesi", "kamera", "máy ảnh", "fotocamera", "aparat|kamera", "กล้อง"),
 ("bluetooth", "蓝牙|藍牙", "", "", "بلوتوث", "блютуз|bluetooth", "ブルートゥース", "블루투스", "", "", "ब्लूटूथ", "", "", "", "", "", "บลูทูธ"),
 ("wireless", "无线", "inalámbrico|inalámbricos|inalámbrica", "sem fio", "لاسلكي|لاسلكية", "беспроводной|беспроводные|беспроводная", "ワイヤレス|無線", "무선", "kabellos|drahtlos", "sans fil", "वायरलेस", "kablosuz", "nirkabel", "không dây", "wireless|senza fili", "bezprzewodowy|bezprzewodowe|bezprzewodowa", "ไร้สาย"),
 # ---- home & beauty
 ("cup", "杯子|杯", "taza|vaso", "copo|caneca", "كوب", "чашка|кружка|стакан", "カップ|コップ", "컵", "tasse|becher", "tasse|gobelet", "कप", "bardak|fincan", "cangkir|gelas", "cốc|ly", "tazza|bicchiere", "kubek", "แก้ว|ถ้วย"),
 ("bottle", "瓶|水瓶", "botella", "garrafa", "زجاجة|قارورة", "бутылка", "ボトル", "병|물병", "flasche", "bouteille", "बोतल", "şişe", "botol", "chai", "bottiglia", "butelka", "ขวด"),
 ("lamp", "灯|台灯", "lámpara", "luminária|lâmpada", "مصباح", "лампа", "ランプ|ライト", "램프|조명", "lampe", "lampe", "लैंप", "lamba", "lampu", "đèn", "lampada", "lampa", "โคมไฟ"),
 ("towel", "毛巾", "toalla", "toalha", "منشفة", "полотенце", "タオル", "수건", "handtuch", "serviette", "तौलिया", "havlu", "handuk", "khăn tắm|khăn", "asciugamano", "ręcznik", "ผ้าเช็ดตัว"),
 ("toy", "玩具", "juguete|juguetes", "brinquedo|brinquedos", "لعبة|ألعاب", "игрушка|игрушки", "おもちゃ|玩具", "장난감", "spielzeug", "jouet|jouets", "खिलौना", "oyuncak", "mainan", "đồ chơi", "giocattolo|giocattoli", "zabawka|zabawki", "ของเล่น"),
 ("book", "书", "libro", "livro", "كتاب", "книга", "本", "책", "buch", "livre", "किताब|पुस्तक", "kitap", "buku", "sách", "libro", "książka", "หนังสือ"),
 ("perfume", "香水", "perfume", "perfume", "عطر", "духи|парфюм", "香水", "향수", "parfüm", "parfum", "इत्र|परफ्यूम", "parfüm", "parfum", "nước hoa", "profumo", "perfumy", "น้ำหอม"),
 ("cream", "霜|乳霜", "crema", "creme", "كريم", "крем", "クリーム", "크림", "creme", "crème", "क्रीम", "krem", "krim", "kem", "crema", "krem", "ครีม"),
 ("shampoo", "洗发水", "champú", "xampu", "شامبو", "шампунь", "シャンプー", "샴푸", "shampoo", "shampooing", "शैम्पू", "şampuan", "sampo", "dầu gội", "shampoo", "szampon", "แชมพู"),
 # ---- materials
 ("leather", "皮|皮革|真皮", "cuero|piel", "couro", "جلد|جلدي|جلدية", "кожаная|кожаный|кожаные|кожа", "レザー|革|本革", "가죽", "leder", "cuir", "चमड़ा|चमड़े", "deri", "kulit", "da", "pelle|cuoio", "skórzana|skórzany|skórzane|skóra", "หนัง"),
 ("cotton", "棉|纯棉", "algodón", "algodão", "قطن|قطني", "хлопковая|хлопковый|хлопок", "綿|コットン", "면|코튼", "baumwolle", "coton", "सूती|कपास", "pamuk|pamuklu", "katun", "cotton|vải bông", "cotone", "bawełniana|bawełniany|bawełna", "ผ้าฝ้าย|ฝ้าย"),
 ("silk", "丝绸|真丝", "seda", "seda", "حرير", "шёлк|шелковый|шелковая", "シルク|絹", "실크", "seide", "soie", "रेशम", "ipek", "sutra", "lụa", "seta", "jedwab|jedwabny", "ผ้าไหม"),
 ("wool", "羊毛", "lana", "lã", "صوف", "шерсть|шерстяной|шерстяная", "ウール|羊毛", "울|양모", "wolle", "laine", "ऊन|ऊनी", "yün", "wol", "len", "lana", "wełna|wełniany", "ขนสัตว์"),
 ("plastic", "塑料", "plástico", "plástico", "بلاستيك", "пластик|пластиковый|пластиковая", "プラスチック", "플라스틱", "kunststoff|plastik", "plastique", "प्लास्टिक", "plastik", "plastik", "nhựa", "plastica", "plastik|plastikowy", "พลาสติก"),
 ("stainless steel", "不锈钢", "acero inoxidable", "aço inoxidável", "ستانلس ستيل|فولاذ مقاوم للصدأ", "нержавеющая сталь|нержавейка", "ステンレス", "스테인리스", "edelstahl", "acier inoxydable|inox", "स्टेनलेस स्टील", "paslanmaz çelik", "stainless steel|baja tahan karat", "thép không gỉ|inox", "acciaio inossidabile|acciaio inox", "stal nierdzewna", "สแตนเลส"),
 ("wood", "木|木质|实木", "madera", "madeira", "خشب|خشبي", "дерево|деревянный|деревянная", "木|木製", "나무|원목", "holz", "bois", "लकड़ी", "ahşap", "kayu", "gỗ", "legno", "drewno|drewniany", "ไม้"),
 ("glass", "玻璃", "vidrio|cristal", "vidro", "زجاج", "стекло|стеклянный", "ガラス", "유리", "glas", "verre", "कांच", "cam", "kaca", "thủy tinh|kính", "vetro", "szkło|szklany", "แก้ว|กระจก"),
 ("gold", "金|黄金", "oro", "ouro", "ذهب|ذهبي", "золото|золотой|золотая", "ゴールド|金", "금|골드", "gold", "or", "सोना|सोने", "altın", "emas", "vàng", "oro", "złoto|złoty", "ทอง"),
 ("silver", "银|纯银", "plata", "prata", "فضة|فضي", "серебро|серебряный|серебряная", "シルバー|銀", "은|실버", "silber", "argent", "चाँदी", "gümüş", "perak", "bạc", "argento", "srebro|srebrny", "เงิน"),
 # ---- colors
 ("black", "黑色|黑", "negro|negra|negros|negras", "preto|preta|pretos|pretas", "أسود|سوداء", "чёрный|черный|чёрная|черная|чёрные|черные", "黒|ブラック", "검은색|블랙|검정", "schwarz|schwarze|schwarzer|schwarzes", "noir|noire|noirs|noires", "काला|काले|काली", "siyah", "hitam", "đen|màu đen", "nero|nera|neri|nere", "czarny|czarna|czarne", "สีดำ|ดำ"),
 ("white", "白色|白", "blanco|blanca|blancos|blancas", "branco|branca|brancos|brancas", "أبيض|بيضاء", "белый|белая|белые", "白|ホワイト", "흰색|화이트|하얀", "weiß|weiss|weiße|weißer", "blanc|blanche|blancs|blanches", "सफ़ेद|सफेद", "beyaz", "putih", "trắng|màu trắng", "bianco|bianca|bianchi|bianche", "biały|biała|białe", "สีขาว|ขาว"),
 ("red", "红色|红", "rojo|roja|rojos|rojas", "vermelho|vermelha", "أحمر|حمراء", "красный|красная|красные", "赤|レッド", "빨간색|레드|빨강", "rot|rote|roter", "rouge|rouges", "लाल", "kırmızı", "merah", "đỏ|màu đỏ", "rosso|rossa|rossi", "czerwony|czerwona|czerwone", "สีแดง|แดง"),
 ("blue", "蓝色|蓝", "azul|azules", "azul", "أزرق|زرقاء", "синий|синяя|синие|голубой", "青|ブルー", "파란색|블루|파랑", "blau|blaue|blauer", "bleu|bleue|bleus", "नीला|नीले", "mavi", "biru", "xanh dương|màu xanh", "blu|azzurro", "niebieski|niebieska|niebieskie", "สีน้ำเงิน|สีฟ้า"),
 ("green", "绿色|绿", "verde|verdes", "verde", "أخضر|خضراء", "зелёный|зеленый|зелёная|зеленая", "緑|グリーン", "초록색|그린|녹색", "grün|grüne", "vert|verte", "हरा|हरे", "yeşil", "hijau", "xanh lá", "verde", "zielony|zielona|zielone", "สีเขียว"),
 ("pink", "粉色|粉红色", "rosa", "rosa", "وردي", "розовый|розовая", "ピンク", "분홍색|핑크", "rosa", "rose", "गुलाबी", "pembe", "merah muda|pink", "hồng", "rosa", "różowy|różowa", "สีชมพู"),
 ("grey", "灰色", "gris", "cinza", "رمادي", "серый|серая", "グレー|灰色", "회색|그레이", "grau", "gris|grise", "स्लेटी|ग्रे", "gri", "abu-abu", "xám", "grigio|grigia", "szary|szara", "สีเทา"),
 ("brown", "棕色|咖啡色", "marrón|café", "marrom", "بني", "коричневый|коричневая", "茶色|ブラウン", "갈색|브라운", "braun", "marron", "भूरा", "kahverengi", "coklat", "nâu", "marrone", "brązowy|brązowa", "สีน้ำตาล"),
 ("yellow", "黄色", "amarillo|amarilla", "amarelo|amarela", "أصفر|صفراء", "жёлтый|желтый", "黄色|イエロー", "노란색|옐로우", "gelb", "jaune", "पीला", "sarı", "kuning", "vàng", "giallo|gialla", "żółty|żółta", "สีเหลือง"),
 # ---- sizes / quantities / misc
 ("size", "尺码|尺寸|码", "talla|tamaño", "tamanho", "مقاس|حجم", "размер", "サイズ", "사이즈|크기", "größe", "taille", "साइज़|आकार", "beden|boyut", "ukuran", "kích thước|cỡ|size", "taglia|misura", "rozmiar", "ขนาด|ไซซ์"),
 ("small", "小号|小", "pequeño|pequeña", "pequeno|pequena", "صغير", "маленький|маленькая", "小|スモール", "소형|작은", "klein", "petit|petite", "छोटा", "küçük", "kecil", "nhỏ", "piccolo|piccola", "mały|mała", "เล็ก"),
 ("large", "大号|大", "grande", "grande", "كبير", "большой|большая", "大|ラージ", "대형|큰", "groß|gross", "grand|grande", "बड़ा", "büyük", "besar", "lớn", "grande", "duży|duża", "ใหญ่"),
 ("set", "套装|套", "juego|set|conjunto", "conjunto|kit", "طقم|مجموعة", "набор|комплект", "セット", "세트", "set", "ensemble|lot", "सेट", "set|takım", "set", "bộ", "set", "zestaw|komplet", "ชุด"),
 ("pieces", "件|个|只", "piezas|unidades|uds", "peças|unidades", "قطع|قطعة", "шт|штук|штуки", "個|枚|本", "개", "stück|stk", "pièces|pcs", "पीस", "adet", "pcs|buah", "cái|chiếc", "pezzi|pz", "szt|sztuk", "ชิ้น"),
 ("new", "新|新款", "nuevo|nueva", "novo|nova", "جديد|جديدة", "новый|новая|новые", "新品|新しい", "새|신상", "neu|neue", "nouveau|nouvelle", "नया|नई", "yeni", "baru", "mới", "nuovo|nuova", "nowy|nowa|nowe", "ใหม่"),
 ("original", "正品|原装", "original", "original", "أصلي", "оригинал|оригинальный", "正規品|純正", "정품", "original", "original|d'origine", "असली|ओरिजिनल", "orijinal", "asli|original", "chính hãng", "originale", "oryginalny|oryginał", "ของแท้"),
 ("waterproof", "防水", "impermeable", "impermeável|à prova d'água", "مقاوم للماء", "водонепроницаемый|непромокаемый", "防水", "방수", "wasserdicht", "imperméable|étanche", "वाटरप्रूफ", "su geçirmez", "tahan air", "chống nước", "impermeabile", "wodoodporny", "กันน้ำ"),
 ("with", "带|配", "con", "com", "مع", "с|со", "付き", "포함", "mit", "avec", "के साथ", "ile|li|lı", "dengan", "có|với", "con", "z|ze", "พร้อม"),
 ("for", "用|适用于", "para", "para", "ل|لـ", "для", "用", "용", "für", "pour", "के लिए", "için", "untuk", "cho", "per", "do|dla", "สำหรับ"),
 ("and", "和|与", "y|e", "e", "و", "и", "と", "및|와|과", "und", "et", "और", "ve", "dan", "và", "e|ed", "i|oraz", "และ"),
 ("gift", "礼物|礼品", "regalo", "presente", "هدية", "подарок", "ギフト|プレゼント", "선물", "geschenk", "cadeau", "उपहार|तोहफ़ा", "hediye", "hadiah", "quà tặng|quà", "regalo", "prezent", "ของขวัญ"),
 ("box", "盒子|盒", "caja", "caixa", "صندوق|علبة", "коробка", "箱|ボックス", "상자|박스", "box|schachtel|karton", "boîte", "डिब्बा", "kutu", "kotak", "hộp", "scatola", "pudełko", "กล่อง"),
 ("documents", "文件", "documentos", "documentos", "وثائق|مستندات", "документы", "書類", "서류", "dokumente", "documents", "दस्तावेज़", "belgeler", "dokumen", "tài liệu", "documenti", "dokumenty", "เอกสาร"),
 ("clothes", "衣服|服装", "ropa", "roupas|roupa", "ملابس", "одежда", "衣類|服", "옷|의류", "kleidung", "vêtements", "कपड़े", "kıyafet|giysi", "pakaian|baju", "quần áo", "abbigliamento|vestiti", "ubrania|odzież", "เสื้อผ้า"),
 ("food", "食品", "comida|alimentos", "comida|alimentos", "طعام|أغذية", "еда|продукты", "食品", "식품|음식", "lebensmittel", "nourriture|alimentaire", "भोजन|खाद्य", "gıda|yiyecek", "makanan", "thực phẩm", "cibo|alimenti", "żywność|jedzenie", "อาหาร"),
 ("cosmetics", "化妆品", "cosméticos", "cosméticos", "مستحضرات تجميل", "косметика", "化粧品", "화장품", "kosmetik", "cosmétiques", "सौंदर्य प्रसाधन", "kozmetik", "kosmetik", "mỹ phẩm", "cosmetici", "kosmetyki", "เครื่องสำอาง"),
 ("spare parts", "配件|零件", "repuestos", "peças de reposição", "قطع غيار", "запчасти", "部品|パーツ", "부품", "ersatzteile", "pièces détachées", "स्पेयर पार्ट्स", "yedek parça", "suku cadang", "phụ tùng", "ricambi", "części zamienne", "อะไหล่"),
 ("fragile", "易碎", "frágil", "frágil", "قابل للكسر", "хрупкое|хрупкий", "割れ物|壊れ物", "깨지기 쉬움|파손주의", "zerbrechlich", "fragile", "नाजुक", "kırılabilir|kırılacak", "mudah pecah", "dễ vỡ", "fragile", "kruche|ostrożnie", "แตกง่าย"),
]

# English adjective order for product phrases: lower rank comes first
CATEGORY = {
    "gender": ["women's", "men's", "children's", "baby", "unisex"],
    "quality": ["new", "original", "waterproof", "wireless", "bluetooth", "fragile"],
    "size": ["small", "large"],
    "color": ["black", "white", "red", "blue", "green", "pink", "grey", "brown", "yellow"],
    "material": ["leather", "cotton", "silk", "wool", "plastic", "stainless steel", "wood", "glass", "gold", "silver"],
}
RANK = {"gender": 0, "quality": 1, "size": 2, "color": 3, "material": 4, "item": 5, "function": 6}

# function words dropped inside product phrases (only when the phrase has glossary hits)
FUNCTION_WORDS = {
    "es": "de del la el los las un una al", "pt": "de do da dos das o a os as um uma", "fr": "de du des la le les un une en d",
    "it": "di del della dei delle il la lo le gli un una in", "de": "der die das ein eine aus", "pl": "",
    "tr": "", "id": "yang", "vi": "của", "ru": "", "en": "the a an of",
}
