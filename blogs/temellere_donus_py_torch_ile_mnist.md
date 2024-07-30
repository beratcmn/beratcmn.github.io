# Temellere Dönüş, PyTorch ile MNIST

Selamlar, uzun süredir LLM'ler ve Transformer modelleri üzerine çalışıyorum, bu konuda birçok farklı modeli inceledim ve kendi modellerimi de geliştirdim. Bu süreçte, model geliştirme sürecinde en çok ihtiyaç duyduğum şeylerden biri, modelin nasıl çalıştığını daha iyi anlamak ve modelin içinde neler olduğunu daha iyi görebilmekti. Daha önce temel bir CS (Computer Science) veya ML (Machine Learning) eğitimi almadım, kariyerimde '[self-taught](https://en.wikipedia.org/wiki/Autodidacticism)' ilerliyorum. Sorumluluklarım genelde [LLM](https://en.wikipedia.org/wiki/Large_language_model)'leri fine-tune etmek ve yeni modeller geliştirmek üzerine olduğu için temel ML (Machine Learning) konseptlerinden uzaklaşmaya başladım. Keras ve Tensorflow ile uzun dönem çalışma fırsatı buldum ancak son zamanlarda kendime yaratabildiğim vakitte 'endüstri standartı' haline gelmiş [PyTorch](https://pytorch.org/)'a vakit ayırmak istedim.

## Neler yapacağız?

Bu yazıda, PyTorch ile MNIST veri kümesi üzerinde bir model eğitmek ve bu süreçte PyTorch'u daha iyi anlamak istiyorum.

## Peki PyTorch nedir?

En basit haliyle PyTorch bir "[tensor](<https://en.wikipedia.org/wiki/Tensor_(machine_learning)>) kütüphanesi". Dolayısıyla temel işlevi tensorlar (çok boyutlu array'ler) ile işlemler yapabilmesi.

Buyurun bakalım ChatGPT ne demiş:

PyTorch, açık kaynaklı bir makine öğrenimi kütüphanesidir ve özellikle derin öğrenme modellerinin geliştirilmesi ve eğitilmesi için kullanılır. 2016 yılında Facebook'un AI Research (FAIR) ekibi tarafından geliştirilmiştir. PyTorch, dinamik hesap grafiği ve sezgisel API tasarımı ile bilinir, bu da araştırmacılar ve geliştiriciler tarafından hızlı bir şekilde model prototipleri oluşturmayı ve deney yapmayı kolaylaştırır.

Başlıca özellikleri şunlardır:

1 _Dinamik Hesap Grafiği_: PyTorch, veri akışı sırasında hesap grafiğini dinamik olarak oluşturur. Bu, hata ayıklamayı ve model tasarımını daha kolay ve esnek hale getirir.

2 _Otomatik Diferansiyasyon_: PyTorch, otomatik türev hesaplama (autograd) özelliği ile gradientleri hesaplamayı ve geri yayılımı (backpropagation) otomatikleştirir.

3 _Desteklenen Donanım_: Hem CPU hem de GPU üzerinde çalışabilir, bu da büyük ölçekli hesaplamaları hızlandırır.

4 _Geniş Kapsamlı Kullanım Alanları_: Bilgisayarlı görü, doğal dil işleme, sinir ağları gibi çeşitli alanlarda kullanılabilir.

5 _Gelişmiş Topluluk ve Dokümantasyon_: PyTorch'un güçlü bir topluluğu ve geniş kapsamlı dokümantasyonu vardır, bu da öğrenme sürecini ve problem çözmeyi kolaylaştırır.

Bu özellikleri sayesinde PyTorch, akademik araştırmalarda ve endüstriyel uygulamalarda yaygın olarak kullanılan bir kütüphane haline gelmiştir.

## Ee MNIST ne?

MNIST, el yazısı rakamlardan oluşan bir veri kümesidir ve genellikle makine öğrenimi ve derin öğrenme modellerinin eğitiminde kullanılır. MNIST veri kümesi, 0 ile 9 arasındaki rakamları temsil eden 28x28 boyutunda siyah-beyaz görüntülerden oluşur. Her bir görüntü, hangi rakamı temsil ettiğini belirten bir etiketle eşlenir. MNIST veri kümesi, genellikle basit bir veri kümesi olarak kabul edilir ve yeni bir makine öğrenimi modeli oluştururken ilk adım olarak sıkça kullanılır.

Kısaca ML (Machine Learning) dünyasında bir 'Hello World' olarak kabul edilir.

## Hadi başlayalım

### Kurulum

Önce bir virtual environment oluşturup PyTorch kütüphanesini yükleyelim:

PyTorch dökümantasyonuna [buradan](https://pytorch.org/get-started/locally/) ulaşabilirsiniz.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio
```

(Opsiyonel) Ek olarak Keras'tan alışkın olduğum model özetlerini görebilmek için torchinfo kütüphanesini de yükleyelim:

```bash
pip install torchinfo
```

Bu süreçte VSCode kullanacağım ve Jupyter Notebook'u VSCode üzerinde çalıştıracağım.

### Hadi kütüphaneleri import edelim

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
```

### Transformları tanımlayalım

```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
```

Transformlar nedir? Transformlar, veri kümesindeki görüntüler üzerinde çeşitli işlemler yapmamızı sağlar. Örneğin, ToTensor() metodu, görüntüyü tensora dönüştürür. Normalize() metodu, görüntü piksellerini belirli bir ortalama ve standart sapma ile normalize eder. Bu işlemler, veri kümesinin daha iyi eğitilmesine ve modelin daha iyi performans göstermesine yardımcı olur çoğunlukla.

Kısaca: Verimizi standart bir hale getirmeye çalışıyoruz.

### Veri kümesini yükleyelim

```python
# Train
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Test
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
```

**trainset** ile eğitim veri kümesini, **testset** ile test veri kümesini yüklüyoruz. **trainloader** ve **testloader** ile de bu veri kümelerini daha rahat kullanabilmek için DataLoader'a yüklüyoruz. Çok düzgün bir açıklama olmadı bu ama anladığım kadarıyla PyTorch'da veriler **DataLoader**'lar ile yükleniyor ve bu DataLoader'lar üzerinden veri kümesine erişim sağlanıyor.

**batch_size** parametresi, her bir adımda kaç verinin işleneceğini belirler. **shuffle** parametresi ise veri kümesinin her bir epoch'ta karıştırılıp karıştırılmayacağını belirler.

### Modeli tanımlayalım

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

Adım adım gidelim:

1. **Net** adında bir sınıf tanımlıyoruz ve bu sınıf **nn.Module**'den türetiliyor.
2. **\_\_init\_\_** metodu, sınıfın başlatıcı metodu olup, modelin katmanlarını tanımlar.
3. **forward** metodu, modelin ileri geçişini tanımlar. İleri geçiş, modelin girdisini alır ve çıktıyı üretir.
4. **nn.Linear** metodu, tam bağlı bir katman oluşturur. İ
5. **F.relu** metodu, ReLU aktivasyon fonksiyonunu uygular.
6. **x.view(-1, 28 \* 28)**, girdiyi yeniden şekillendirir. -1, girdinin boyutunu otomatik olarak ayarlar.
7. **x = F.relu(self.fc1(x))**, girdiyi ilk tam bağlı katmandan geçirir ve ReLU aktivasyon fonksiyonunu uygular.
8. **x = F.relu(self.fc2(x))**, girdiyi ikinci tam bağlı katmandan geçirir ve ReLU aktivasyon fonksiyonunu uygular.
9. **x = self.fc3(x)**, girdiyi üçüncü tam bağlı katmandan geçirir ve çıktıyı üretir.
10. **return x**, çıktıyı döndürür.

Burada model parametrelerini kafama göre belirledim, daha iyi bir model için parametre optimizasyonu yapılabilir. Bunlar biraz daha deneme yanılma ile öğrenilecek şeyler bana kalırsa.

### Modeli, Loss fonksiyonunu ve Optimizer'ı tanımlayalım

```python
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
```

1. **net = Net()**, modeli oluşturur.
2. **criterion = nn.CrossEntropyLoss()**, loss fonksiyonunu tanımlar. CrossEntropyLoss, sınıflandırma problemleri için yaygın olarak kullanılan bir loss fonksiyonudur.
3. **optimizer = torch.optim.Adam(net.parameters(), lr=0.001)**, optimizer'ı tanımlar. Adam optimizer, gradient tabanlı optimizasyon algoritmalarından biridir.

Burada da Loss fonksiyonunu ve Optimizer'ı tecrübeme dayanarak belirledim. Daha iyi bir model için bu parametrelerin değiştirilmesi gerekebilir. Örneğin Adam optimizer yerine SGD optimizer kullanılabilir.

### Model özetini görelim

```python
summary(net, input_size=(64, 1, 28, 28))
```

Model özetini görmek için **torchinfo** kütüphanesini kullandım. Bu kütüphane, modelin katmanlarını ve parametrelerini gösterir. Burada **inputsize** parametresi, modelin girdi boyutunu belirtir. (batch_size, channels, height, width)

Model özeti aşağıdaki gibi olacak:

```bash
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Net                                      [64, 10]                  --
├─Linear: 1-1                            [64, 512]                 401,920
├─Linear: 1-2                            [64, 256]                 131,328
├─Linear: 1-3                            [64, 10]                  2,570
==========================================================================================
Total params: 535,818
Trainable params: 535,818
Non-trainable params: 0
Total mult-adds (M): 34.29
==========================================================================================
Input size (MB): 0.20
Forward/backward pass size (MB): 0.40
Params size (MB): 2.14
Estimated Total Size (MB): 2.74
==========================================================================================
```

### Modeli eğitelim

```python
for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')
```

1. **for epoch in range(5)**, 5 epoch boyunca eğitim veri kümesi üzerinde döngü oluşturur.
2. **running_loss = 0.0**, loss değerini sıfırlar.
3. **for i, data in enumerate(trainloader, 0)**, eğitim veri kümesinde döngü oluşturur.
4. **inputs, labels = data**, veri kümesinden girdi ve etiketleri alır.
5. **optimizer.zero_grad()**, gradyanları sıfırlar.
6. **outputs = net(inputs)**, modeli eğitir.
7. **loss = criterion(outputs, labels)**, loss değerini hesaplar.
8. **loss.backward()**, gradyanları hesaplar.
9. **optimizer.step()**, modeli günceller.
10. **running_loss += loss.item()**, loss değerini toplar.
11. **if i % 100 == 99**, her 100 mini-batch'te loss değerini yazdırır.
12. **print('Finished Training')**, eğitimi tamamlar.

Burada Keras'a göre biraz daha fazla kod yazdım. PyTorch'da _training loop_ yazarken daha fazla kontrol sahibi oluyorsunuz. Bu, modelin nasıl eğitildiğini daha iyi anlamanıza yardımcı olabilir. Biraz daha proje yapmaya ihtiyacım var.

### Modeli test edelim

```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
```

1. **correct = 0, total = 0**, doğru tahmin sayısını ve toplam tahmin sayısını sıfırlar.
2. **with torch.no_grad()**, gradyanları kapatır. Bu, modelin eğitim sırasında gradyanları güncellememesini sağlar.
3. **for data in testloader**, test veri kümesinde döngü oluşturur.
4. **images, labels = data**, veri kümesinden girdi ve etiketleri alır.
5. **outputs = net(images)**, modeli test eder.
6. **\_, predicted = torch.max(outputs.data, 1)**, tahminleri alır.
7. **total += labels.size(0)**, toplam tahmin sayısını artırır.
8. **correct += (predicted == labels).sum().item()**, doğru tahmin sayısını artırır.
9. **print(f'Accuracy of the network on the 10000 test images: {100 \* correct / total:.2f}%')**, doğruluk oranını yazdırır.

## Sonuç

ML tarafında yapabileceğimiz en basit projelerden biri buydu, PyTorch'u çok daha iyi anladığımı düşünüyorum ancak yazdığım kodun daha düzgün olması adına ChatGPT ve Stackoverflow gibi kaynaklardan destek aldım, ne yazık ki geçmiş tecrübemi göz önünde bulundurunca benim için öğrenme konusunda bir tık olumsuz bir tecrübe oldu. Ancak yine de PyTorch'u daha iyi anlamak için bu tarz basit projelerin yapılması gerektiğini düşünüyorum. Umarım bu yazı, PyTorch'u daha iyi anlamak isteyenlere yardımcı olur.

Projenin olduğu Github reposuna [buradan](https://github.com/beratcmn/pytorch-mnist) ulaşabilirsiniz.

Ayrıca bu blok yazısını yazabilmek için web siteme sıfırdan blog mekanizması eklemem gerekti 😅, bunu da dökümante etmek istiyorum.

Görüşmek üzere!