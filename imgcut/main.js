var postFile = {
  init: function () {
    var t = this;
    t.regional = document.getElementById('label');
    t.getImage = document.getElementById('get_image');
    t.editPic = document.getElementById('edit_pic');
    t.editBox = document.getElementById('cover_box');
    t.px = 0; //background image x
    t.py = 0; //background image y
    t.sx = 15; //crop area x
    t.sy = 15; //crop area y
    t.sHeight = 100; //crop area height
    t.sWidth = 100; //crop area width
    document
      .getElementById('post_file')
      .addEventListener('change', t.handleFiles, false);

    document.getElementById('save_button').onclick = function () {
      t.editPic.height = t.sHeight;
      t.editPic.width = t.sWidth;
      var ctx = t.editPic.getContext('2d');
      var images = new Image();
      images.src = t.imgUrl;

      images.onload = function () {
        ctx.drawImage(
          images,
          t.sx,
          t.sy,
          t.sHeight,
          t.sWidth,
          0,
          0,
          t.sHeight,
          t.sWidth
        );
        document
          .getElementById('show_pic')
          .getElementsByTagName('img')[0].src = t.editPic.toDataURL();
      };
    };
  },

  handleFiles: function () {
    var fileList = this.files[0];
    var oFReader = new FileReader();
    oFReader.readAsDataURL(fileList);
    oFReader.onload = function (oFREvent) {
      postFile.paintImage(oFREvent.target.result);
    };
  },

  paintImage: function (url) {
    var t = this;
    var createCanvas = t.getImage.getContext('2d');
    var img = new Image();
    img.src = url;
    img.onload = function () {
      if (
        img.width < t.regional.offsetWidth &&
        img.height < t.regional.offsetHeight
      ) {
        t.imgWidth = img.width;
        t.imgHeight = img.height;
      } else {
        var pWidth = img.width / (img.height / t.regional.offsetHeight);
        var pHeight = img.height / (img.width / t.regional.offsetWidth);
        t.imgWidth = img.width > img.height ? t.regional.offsetWidth : pWidth;
        t.imgHeight =
          img.height > img.width ? t.regional.offsetHeight : pHeight;
      }
      t.px = (t.regional.offsetWidth - t.imgWidth) / 2 + 'px';
      t.py = (t.regional.offsetHeight - t.imgHeight) / 2 + 'px';

      t.getImage.height = t.imgHeight;
      t.getImage.width = t.imgWidth;
      t.getImage.style.left = t.px;
      t.getImage.style.top = t.py;

      createCanvas.drawImage(img, 0, 0, t.imgWidth, t.imgHeight);
      t.imgUrl = t.getImage.toDataURL();
      t.cutImage();
      t.drag();
    };
  },



postFile.init();
