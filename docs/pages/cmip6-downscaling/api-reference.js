import React from 'react'
import { promises as fs } from 'fs'
import { Box, Themed } from 'theme-ui'
import Themify from '../../components/themify'
import Section from '../../components/section'

const prefix = 'cmip6-downscaling'

const APIReference = ({ body }) => {
  body = body.replace(/..\/generated/g, `../${prefix}/generated`)
  return (
    <Box>
      <Section name='API'>
        <Themed.h1>API reference</Themed.h1>
        <Themify html={body} />
      </Section>
    </Box>
  )
}

export default APIReference

export async function getStaticProps({ params }) {
  const res = await fs.readFile('./_build/json/api.fjson', 'utf8')
  const contents = JSON.parse(res)
  return { props: contents }
}
