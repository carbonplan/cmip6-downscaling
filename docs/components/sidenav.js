import { Box } from 'theme-ui'
import { Link } from '@carbonplan/components'
import { contents } from './contents'

const Sidenav = ({ active, expanded }) => {
  return (
    <>
      <Box
        sx={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          opacity: expanded ? 0.9 : 0,
          transition: 'opacity 0.15s',
          display: ['initial', 'initial', 'none', 'none'],
          bg: ['background', 'background', 'transparent', 'transparent'],
          zIndex: [500, 500, 'initial', 'initial'],
          pointerEvents: expanded ? 'all' : 'none',
        }}
      />
      <Box
        sx={{
          fontSize: [3, 3, 3, 4],
          mb: [8, 8, 9, 10],
          overflow: 'scroll',
          position: ['fixed', 'fixed', 'sticky', 'sticky'],
          height: [
            'calc(100vh - 56px)',
            'calc(100vh - 56px)',
            'calc(100vh - 128px)',
            'calc(100vh - 128px)',
          ],
          marginTop: ['56px', '56px', '104px', '114px'],
          top: ['0px', '0px', '160px', '170px'],
          pt: [5, 6, 0, 0],
          pr: [5, 6, 0, 0],
          transform: [
            expanded ? 'none' : 'translateX(-125%)',
            expanded ? 'none' : 'translateX(-120%)',
            'none',
            'none',
          ],
          transition: [
            'transform 0.15s ease',
            'transform 0.15s ease',
            'none',
            'none',
          ],
          width: [
            'calc(4/6 * 100vw - 28px)',
            'calc(3/8 * 100vw - 40px)',
            'auto',
            'auto',
          ],
          bg: ['background', 'background', 'transparent', 'transparent'],
          zIndex: [1000, 1000, 'initial', 'initial'],
          borderRight: ({ colors }) =>
            expanded ? `solid 1px ${colors.muted}` : 'none',
        }}
      >
        <Link
          href={'/'}
          sx={{
            width: 'fit-content',
            display: 'block',
            textDecoration: 'none',
            color: 'intro' === active ? 'primary' : 'secondary',
            '&:hover': {
              color: 'primary',
            },
          }}
        >
          Intro
        </Link>
        {Object.keys(contents).map((d) => {
          return (
            <Box key={d}>
              <Box
                sx={{
                  pt: [2],
                  color: 'secondary',
                  fontSize: [2, 2, 2, 3],
                  letterSpacing: 'smallcaps',
                  fontFamily: 'heading',
                  textTransform: 'uppercase',
                  mt: [3],
                }}
              >
                {d}
              </Box>
              <Box sx={{ my: [2] }}>
                {contents[d].map((e) => {
                  const href =
                    '/' +
                    (e['href'] ? e['href'] : e.replace(/ /g, '-').toLowerCase())
                  const label = e['label'] ? e['label'] : e
                  return (
                    <Link
                      key={label}
                      href={href}
                      sx={{
                        width: 'fit-content',
                        display: 'block',
                        textDecoration: 'none',
                        color:
                          label.toLowerCase() === active
                            ? 'primary'
                            : 'secondary',
                        '&:hover': {
                          color: 'primary',
                        },
                      }}
                    >
                      {label}
                    </Link>
                  )
                })}
              </Box>
            </Box>
          )
        })}
      </Box>
    </>
  )
}

export default Sidenav
