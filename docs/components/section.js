import { useState } from 'react'
import { Layout, Row, Column, FadeIn } from '@carbonplan/components'
import Sidenav from './sidenav'

const Section = ({ children, name }) => {
  const [expanded, setExpanded] = useState(false)

  return (
    <Layout
      fade={false}
      settings={{
        value: expanded,
        onClick: () => setExpanded((prev) => !prev),
      }}
    >
      <Row>
        <Column start={[1, 1, 2, 2]} width={[4, 4, 2, 2]}>
          <Sidenav active={name} expanded={expanded} />
        </Column>
        <Column start={[1, 2, 5, 5]} width={[6]} sx={{ mb: [8, 8, 9, 10] }}>
          <FadeIn>{children}</FadeIn>
        </Column>
      </Row>
    </Layout>
  )
}

export default Section
