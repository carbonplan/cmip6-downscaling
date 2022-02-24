const isDev =
  process.env.VERCEL_ENV === 'preview' || process.env.NODE_ENV === 'development'

const slug = require('rehype-slug')

const withMDX = require('@next/mdx')({
  extension: /\.mdx?$/,
  options: {
    rehypePlugins: [slug],
  },
})

module.exports = withMDX({
  pageExtensions: ['jsx', 'js', 'md', 'mdx'],
  async rewrites() {
    return [
      {
        source: '/cmip6-downscaling/api',
        destination: '/cmip6-downscaling/api-reference',
      },
    ]
  },
  assetPrefix: isDev ? '' : 'https://cmip6-downscaling.docs.carbonplan.org',
})
