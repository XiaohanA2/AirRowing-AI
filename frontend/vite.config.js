import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'

//http://117.72.111.42:7000/
// https://vitejs.dev/config/
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://117.72.111.42:7000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/file': {
        target: 'http://117.72.111.42:8081', // 新增的图片上传代理
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/file/, '/file'),  // 保留路径
      },
      '/ai': {
        target: 'http://117.72.111.42:5000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/ai/, ''),
      },
    }
  },
  plugins: [
    vue(),
    AutoImport({
      resolvers: [ElementPlusResolver()],
    }),
    Components({
      resolvers: [ElementPlusResolver()],
    }),
  ],
  resolve: {
    alias: {
      // 定义一个别名 '@'，该别名对应于当前 JavaScript 模块文件所在目录下的 'src' 目录的绝对文件路径。
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  }
})
