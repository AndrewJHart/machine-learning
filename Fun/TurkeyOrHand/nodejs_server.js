const http = require('http');

const hostname = '127.0.0.1';
const port = 80;
var formidable = require('formidable');
var fs = require('fs');

var spawn = require('child_process').spawn,
py = spawn('python', ['server_script.py']);

const server = http.createServer((req, res) => {
    switch(req.url) {
        case '/predict':
            var form = new formidable.IncomingForm();
            form.parse(req, function (err, fields, files) {
                var oldpath = files.filetoupload.path;
                var newpath = './' + files.filetoupload.name;
                fs.rename(oldpath, newpath, function (err) {
                    if (err) throw err;
                    py.stdout.on('data', function(data) {
                        res.write("Results: " + data);
                        res.end();
                    });
                    py.stdin.write(files.filetoupload.name);
                    py.stdin.end();
               });
            });
        break;
        case '/':
            res.writeHead(200, {'Content-Type': 'text/html'});
            res.write('<form action="predict" method="post" enctype="multipart/form-data">');
            res.write('<input type="file" name="filetoupload"><br>');
            res.write('<input type="submit">');
            res.write('</form>');
            return res.end();
        break;
    }
});

server.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});
