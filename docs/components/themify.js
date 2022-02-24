import React from 'react'
import h2r from 'html-to-react'
import { RotatingArrow } from '@carbonplan/icons'
import { Row, Column } from '@carbonplan/components'
import { Link, Box, Themed } from 'theme-ui'

const processNode = new h2r.ProcessNodeDefinitions(React)
const parser = new h2r.Parser()

const Span = ({ children }) => <span>{children}</span>
const Noop = () => null

const styleTags = {
  a: Themed.a,
  h2: Themed.h2,
  h3: Themed.h1,
  h4: Themed.h2,
  h5: Themed.h1,
  h6: Themed.h2,
  p: Span,
  code: Span,
  pre: Span,
  ol: Themed.ol,
  ul: Span,
  li: Span,
  hr: Themed.thematicBreak,
  em: Span,
  tr: Span,
  th: Span,
  td: Span,
  strong: Themed.strong,
  del: Themed.del,
  b: Themed.b,
  i: Themed.p,
  inlineCode: Themed.inlineCode,
  dd: Span,
  dt: Span,
}

const styleClasses = {
  h1: Noop,
  headerlink: Noop,
  sig: ({ children }) => (
    <Box
      sx={{
        bg: 'hinted',
        px: [3],
        py: [3],
        fontFamily: 'mono',
        letterSpacing: 'mono',
        fontSize: [1, 1, 1, 2],
        color: 'primary',
        mb: [3, 3, 3, 4],
      }}
    >
      {children}
    </Box>
  ),
  'sig-paren': ({ children }) => (
    <Box as='span' sx={{ color: 'secondary' }}>
      {children}
    </Box>
  ),
  'viewcode-link': Noop,
  classifier: ({ children }) => (
    <Box as='span' sx={{ ml: [2], color: 'secondary' }}>
      {children}
    </Box>
  ),
}

const mappedTags = Object.keys(styleTags)
const mappedClasses = Object.keys(styleClasses)

const instructions = [
  // handle code blocks
  {
    shouldProcessNode: (node) => {
      return node.name === 'code' && node.parent.name === 'pre'
    },
    processNode: (node, children, index) => {
      return <Themed.code key={index}>{children}</Themed.code>
    },
  },
  // remove headings
  {
    shouldProcessNode: (node) => {
      return node.name === 'h1'
    },
    processNode: (node, children, index) => {
      return null
    },
  },
  // make all caps section labels
  {
    shouldProcessNode: (node) => {
      return (
        (node.name === 'dt' &&
          node.attribs?.class &&
          (node.attribs.class === 'field-odd' ||
            node.attribs.class === 'field-even')) ||
        (node.attribs?.class && node.attribs.class === 'rubric')
      )
    },
    processNode: (node, children, index) => {
      return (
        <Box
          key={index}
          sx={{
            mt: [5, 5, 5, 6],
            textTransform: 'uppercase',
            fontFamily: 'heading',
            letterSpacing: 'smallcaps',
          }}
        >
          {children}
        </Box>
      )
    },
  },
  // collapse nested tables
  {
    shouldProcessNode: (node) => {
      return (
        node.name === 'dl' &&
        node.attribs?.class &&
        node.attribs.class === 'simple' &&
        node.parent.name === 'dd' &&
        node.parent.parent.name === 'dl' &&
        node.parent.parent.attribs?.class &&
        node.parent.parent.attribs?.class === 'simple'
      )
    },
    processNode: (node, children, index) => {
      return <Box key={index}>{children}</Box>
    },
  },
  // make tables from simple description lists
  {
    shouldProcessNode: (node) => {
      return (
        node.parent.name === 'dd' &&
        node.name === 'dl' &&
        node.attribs?.class &&
        node.attribs.class === 'simple'
      )
    },
    processNode: (node, children, index) => {
      const indices = Array(Math.round(children.length / 3))
        .fill(0)
        .map((_, d) => 3 * d + 1)

      return (
        <Box key={index}>
          {indices.map((d, i) => {
            return (
              <Row
                key={i}
                columns={6}
                sx={{ wordBreak: 'break-all', mt: [3, 3, 3, 4] }}
              >
                <Column start={1} width={2}>
                  <Box sx={{ fontSize: [1, 1, 1, 2], fontFamily: 'mono' }}>
                    {children[d]}
                  </Box>
                </Column>
                <Column start={3} width={4}>
                  {children[d + 1]}
                </Column>
              </Row>
            )
          })}
        </Box>
      )
    },
  },
  // make tables for other tables
  {
    shouldProcessNode: (node) => {
      return node.name === 'table'
    },
    processNode: (node, children, index) => {
      return children[3].props.children.map((d, i) => {
        if (d.props?.children.length > 0)
          return (
            <Row key={i} columns={6} sx={{ mt: [3, 3, 3, 4] }}>
              <Column start={1} width={3}>
                <Box
                  sx={{
                    fontSize: [1, 1, 1, 2],
                    fontFamily: 'mono',
                    wordBreak: 'break-all',
                  }}
                >
                  {d.props.children[0]}
                </Box>
              </Column>
              <Column start={4} width={3}>
                {d.props.children[2]}
              </Column>
            </Row>
          )
      })
    },
  },
  ...mappedClasses.map((key) => {
    return {
      shouldProcessNode: (node) => {
        if (node.attribs?.class) {
          return (
            node.attribs.class.split(' ').includes(key) ||
            node.attribs.class === key
          )
        }
      },
      processNode: (node, children, index) => {
        const Component = styleClasses[key]
        const props = Object.assign({}, node.attribs)
        delete props.class
        return (
          <Component key={index} {...props}>
            {children}
          </Component>
        )
      },
    }
  }),
  ...mappedTags.map((key) => {
    return {
      shouldProcessNode: (node) => {
        return node.name === key
      },
      processNode: (node, children, index) => {
        const Component = styleTags[key]
        const props = Object.assign({}, node.attribs)
        delete props.class
        return (
          <Component key={index} {...props}>
            {children}
          </Component>
        )
      },
    }
  }),
  {
    shouldProcessNode: () => true,
    processNode: processNode.processDefaultNode,
  },
]

const Themify = ({ html }) => {
  const element = parser.parseWithInstructions(html, () => true, instructions)
  return element
}

export default Themify
