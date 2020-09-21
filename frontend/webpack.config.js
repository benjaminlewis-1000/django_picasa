var path = require('path');
var webpack = require('webpack');
var BundleTracker = require('webpack-bundle-tracker');

module.exports = {
  entry:  path.join(__dirname, 'static/src/js/index'),  

  output: {
    path: path.join(__dirname, 'static/dist'),
    filename: '[name]-[hash].js'
  }, 
 
 plugins: [
    new BundleTracker({
      path: __dirname,
      filename: 'webpack-stats.json'
    }),
  ],

}
