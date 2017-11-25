var spawn = require('child_process').spawn;
var express = require('express');
var util = require("util");
var fs = require("fs");
var fileUpload = require('express-fileupload');
var router = express.Router();

var CATEGORIES = ["Live Turkey", "Cooked Turkey", "Turkey Hand", "Hand"];

router.use(fileUpload({
    safeFileNames: true
}));

/* GET users listing. */
router.get('/', function(req, res) {
  res.send('No file uploaded!');
});

router.post('/', function(req, res, next) {
    if (!req.files)
        return res.status(400).send('No files were uploaded.');

    // The name of the input field (i.e. "sampleFile") is used to retrieve the uploaded file
    let sampleFile = req.files.image_upload;
    filePath = './public/uploads/' + sampleFile.name;

    // Use the mv() method to place the file somewhere on your server
    sampleFile.mv(filePath, function(err) {
        if (err) {
            return res.status(500).send(err);
        }
        py = spawn('python', ['../server_script.py']);
        py.stdout.on('data', function(data) {
            fs.unlinkSync(filePath);
            res.render('pred', { result: CATEGORIES[parseInt(data)] });
        });
        py.stdin.write(filePath);
        py.stdin.end();
    });
});

module.exports = router;
