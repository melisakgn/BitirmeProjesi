from flask import Flask, render_template, request, session
import os
import tensorflow 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import vgg16
from werkzeug.utils import secure_filename
import numpy as np

 

UPLOAD_FOLDER = os.path.join('static','uploads')#static/uploads

#ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
app.secret_key = 'This is your secret key to utilize session in Flask'
 
 
@app.route('/')
def index():
    return render_template("index.html")
 
@app.route('/uploadFile',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        content_img = request.files['contentfile']
        content_filename = secure_filename(content_img.filename) #hatalı isimleri düzeltiyor
        
        style_img = request.files['stylefile']
        style_filename = secure_filename(style_img.filename) #hatalı isimleri düzeltiyor
        
        content_img.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))#staticFiles/uploads/dosyaadı
        style_img.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
     
        session['content_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)#Oturuma kaydedilmesi gereken veriler, sunucuda geçici bir dizinde saklanır.
        session['style_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
        
        img_file_path_1 = session.get('content_img_file_path', None)
        img_file_path_2 = session.get('style_img_file_path', None)
        return render_template("show_image.html", image1 = img_file_path_1 , image2=img_file_path_2)



@app.route('/model',  methods=("POST", "GET"))
def model():
    if request.method == 'POST':
        base_image_path = session.get('content_img_file_path', None)
        style_image_path= session.get('style_img_file_path', None)


        result_prefix = "city_generated"


        #total_variation_weight = 1e-6
        style_weight = 1e-6
        content_weight = 2.5e-8

        width,height = load_img(base_image_path).size
        img_nrows = 400
        img_ncols = int(width * img_nrows / height)
        c  = 3 #kanal sayısı RGB olduğu için 3

        def preprocess_image(image_path):
            # Keras load_img metodu ile image okunur. 
            img = load_img(image_path,target_size=(img_nrows,img_ncols))

            # modele göndermek için arraye çevrilir. 
            img_array = np.array(img)

            # model 1,H,W,C şeklinde data kabul ettiği için bir dimension eklenir. 
            img_array = np.expand_dims(img_array,0)
            
            # VGG19 modeline uygun şekilde preprocess işlemi yapılır. 
            processed_img = vgg16.preprocess_input(img_array) #VGG19, giriş görüntüsü RGB değil, BGR olması gerekir. 
            #O yüzden bu fonk kullanıyorum.
            #Bir nörona gelen girdilerin tamamının pozitif veya tamamının negatif olduğu bir durumu ele alalım. 
            #Bu durumda, geriye yayılım sırasında hesaplanan gradyan ya pozitif ya da negatif olacaktır
            # (girişlerle aynı işaretli). Ve bu nedenle,
            # parametre güncellemeleri yalnızca belirli yönlerle sınırlıdır ve bu da yakınsamayı zorlaştırır . 
            #Sonuç olarak, gradyan güncellemeleri farklı yönlerde çok ileri gidiyor ve bu da optimizasyonu zorlaştırıyor . 
            #Veri kümesi simetrik olduğunda (sıfır ortalama ile) birçok algoritma daha iyi performans gösterir.
            
            img_tensor = tf.convert_to_tensor(processed_img) #diziyi tensöre dönüştürür.Tensorflow'un anlayacağı şekil.
            return img_tensor
        
        def deprocess_image(x):
            # X tensoru H,W,C şekline getirilir. 
            x = x.reshape((img_nrows,img_ncols,3))

            # VGG19 modeline gönderirken 0 merkezli olacak şekilde düzenleme yapmıştık. 
            #(preprocess_input metodunu kullanarak) Bunu eski haline getir 
            #Imagenet veri kümesindeki her bir kanalın ortalamasını ekledik.
            x[:, :, 0] += 103.939
            x[:, :, 1] += 116.779
            x[:, :, 2] += 123.68

            # BGR - RGB çevirmek için 
            x = x[:,:,::-1] # -1 tersine çeviriyor 

            # np.clip metodu gönderilen array içerisinde 0 dan küçük olan değerler için 0
            # 255 den büyük olan değerler için 255 olacak şekilde kırpma işlemi yapar. 
            #Bir aralık verildiğinde, aralığın dışındaki değerler aralık kenarlarına kırpılır. 
            #Örneğin, [0, 1] aralığı belirtilirse, 0'dan küçük değerler 0, 1'den büyük değerler 1 olur.
            x = np.clip(x,0,255).astype("uint8") 
            return x
        
        def gram_matrix(x):
            # H,W,C olan tensoru - > C,H,W şekline çevirelim. 
            x = tf.transpose(x, perm = (2, 0, 1)) 

            # C,H*W şekline rehsape etmek için
            features = tf.reshape(x, (tf.shape(x)[0], -1)) 

            # Korelasyonları(Gram matriks) bulmak  için X*XT işlemi yapıyor
            gram = tf.matmul(features, tf.transpose(features))
            return gram
        
        def style_loss(style,generated):
            # style gram_matriksi hesaplanır. 
            S  = gram_matrix(style)
            # Generate edilen resmin style matriksi hesaplanır. 
            G = gram_matrix(generated)

            
            # makalede belirtilen formul ile hesaplanır. 
            return tf.reduce_sum(tf.square(S - G)) / (4.0 * (c ** 2) * ((img_nrows*img_ncols) ** 2))

        def content_loss(base, generated):
            return tf.reduce_sum(tf.square(generated - base)) / (4.0 * (c*img_nrows*img_ncols))


        def total_variation_loss(content_loss,style_loss,alpha=10,beta = 40):
            return content_loss + style_loss
        
        # VGG19 modelini import ettik
        vgg_model = vgg16.VGG16(include_top=False,weights="imagenet")

        # Tüm layerları ve çıktıları bir dict içerisine atadık. 
        layer_dict = dict([(layer.name,layer.output) for layer in vgg_model.layers])

        # özellik çıkarmada kullanacağımız modeli oluşturduk. 
        feature_extractor = tf.keras.models.Model(vgg_model.input,layer_dict)

        # Style için kullanılacak layerlar
        style_layer_names = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        #Keras'ın sunduğu örneğe göre, stil kaybı fonksiyonunu hesaplamak için her bloğun ilk evrişimini, içerik için ise son bloğun ikinci evrişimi.
        # birinci katmanda yatay, dikey veya çapraz çizgileri algılama ve ikinci katmanda köşeleri algılama vb. gibi basit özellikleri öğrenir.
        #CNN katmanlarına yaklaştıkça görüntünün en iyi özellik temsili bulunur.
        # content için kullanılacak layer
        content_layer_name = "block5_conv3"


        def compute_loss(generated_image,base_image,style_image):
            # bir input tensoru oluşturduk.
            #concat=Tensörleri bir boyut boyunca birleştirir.
            input_tensor = tf.concat([base_image,style_image,generated_image],axis=0)
            # özellik çıkarmada kullanacağımız modele gönderdik. 
            features = feature_extractor(input_tensor)

            # loss değerini tutmak için bir değişken oluşturduk. 
            c_loss = tf.zeros(shape=()) #Tüm öğeleri sıfır olarak ayarlanmış bir tensör oluşturur.
            s_loss = tf.zeros(shape=())
            t_loss = tf.zeros(shape=())


            # Content için kullanılacak layerdaki output değerlerini aldık. 
            layer_features = features[content_layer_name]
            # Bu arraydeki 0. değer base_imagein özellikleri 
            base_image_features = layer_features[0,:,:,:]
            # 2. değer üretilen imagein çıkarılan özellikleri 
            generated_features = layer_features[2,:,:,:]
            # content loss metodu kullanarak c_loss değeri hesaplanır
            c_loss = content_weight*content_loss(base_image_features,generated_features)

            
            # Style için kullanılacak layerlar ile s_loss değerleri hesaplanır .
            for layer_name in style_layer_names:
                # ilgili layerın çıktılarını aldık 
                layer_features = features[layer_name]
                # 1.si style inputunun özellikleri
                style_reference_features = layer_features[1,:,:,:]
                # 2.si generate edilen resmin özellikleri
                generated_features = layer_features[2,:,:,:]

                # ilgili layer için style_loss hesaplanır
                sl = style_loss(style_reference_features,generated_features)
                # s_loss değerine eklenir
                s_loss += (style_weight/len(style_layer_names))*sl

            # total loss metodu ile t_loss hesaplanır
            t_loss = total_variation_loss(c_loss,s_loss)
            return t_loss
        
        x = tf.Variable(2.0, trainable=True)
        with tf.GradientTape() as tape:
            y = x**4

        @tf.function
        def compute_loss_and_grads(generated_image,base_image,style_image):
            with tf.GradientTape() as tape:
                # loss değeri hesaplanır 
                loss = compute_loss(generated_image,base_image,style_image)
            # generated_image de değişiklikler yapılarak loss değeri optimize edilir. 
            grads = tape.gradient(loss,generated_image) # tape bağlamında kaydedilen işlemleri kullanarak gradyanı hesaplamak için kullanılır.
            # loss ve gradient değerleri döndürülür.
            return loss,grads
        
        optimizer = tf.keras.optimizers.Adam(
        tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1.0, decay_steps=100, decay_rate=0.96
        )
        )
        # base_image,style_image modele göndermek için preprocessing işleminden geçirdik. 
        base_image = preprocess_image(base_image_path)
        style_reference_image = preprocess_image(style_image_path)
        generated_image = tf.Variable(preprocess_image(base_image_path))


        iterations = 4000 
        for i in range(1, iterations + 1):
            # loss ve gradient değerleri hesaplıyoruz.
            loss, grads = compute_loss_and_grads(
                generated_image, base_image, style_reference_image
            )
            # optimizer ile generated_image e gradient uygulanır ve generated image 
            # yeniden oluşturulur. 
            optimizer.apply_gradients([(grads, generated_image)])
            
            print("Iteration %d: loss=%.2f" % (i, loss))
            
            if i % 5 == 0:
                
               

                # Üretilen image i save etmek için 
                img = deprocess_image(generated_image.numpy())
                fname = "./static/cikti/"+result_prefix + "_at_iteration_%d.png" % i
                tf.keras.preprocessing.image.save_img(fname, img)
                session['cikti_image'] = fname
                break
    
    img_file_path_1 = session.get('content_img_file_path', None)
    img_file_path_2 = session.get('style_img_file_path', None)
    img_file_path_3 = session.get('cikti_image', None)
    return render_template("show_image.html", image1 = img_file_path_1 , image2=img_file_path_2 , image3=img_file_path_3)





 
if __name__=='__main__':
    app.run(debug = False,host='0.0.0.0')