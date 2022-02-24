import React from 'react'
import path from 'path'
import { promises as fs } from 'fs'
import { Box, Themed } from 'theme-ui'
import Themify from '../components/themify'
import Section from '../components/section'

const Summary = ({ body }) => {
  return (
    <Box>
      <Section name='summary'>
        <Themed.h1>Summary</Themed.h1>
        <Themify html={body} />
      </Section>
    </Box>
  )
}

export default Summary

export async function getStaticProps({ params }) {
  const res = await fs.readFile('./_build/json/api.fjson', 'utf8')
  const contents = JSON.parse(res)
  return { props: contents }
}
