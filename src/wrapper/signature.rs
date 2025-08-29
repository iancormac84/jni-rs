use std::{fmt, str::FromStr};

use combine::{
    between, many, parser, parser::range::recognize, satisfy, skip_many, skip_many1, token,
    ParseError, Parser, RangeStream, StdParseResult, Stream,
};

use crate::errors::*;

/// A primitive java type. These are the things that can be represented without
/// an object.
#[allow(missing_docs)]
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub enum Primitive {
    Boolean, // Z
    Byte,    // B
    Char,    // C
    Double,  // D
    Float,   // F
    Int,     // I
    Long,    // J
    Short,   // S
    Void,    // V
}

impl fmt::Display for Primitive {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Primitive::Boolean => write!(f, "Z"),
            Primitive::Byte => write!(f, "B"),
            Primitive::Char => write!(f, "C"),
            Primitive::Double => write!(f, "D"),
            Primitive::Float => write!(f, "F"),
            Primitive::Int => write!(f, "I"),
            Primitive::Long => write!(f, "J"),
            Primitive::Short => write!(f, "S"),
            Primitive::Void => write!(f, "V"),
        }
    }
}

/// Enum representing any java type in addition to method signatures.
#[allow(missing_docs)]
#[derive(Eq, PartialEq, Debug, Clone)]
pub enum JavaType<'a> {
    Primitive(Primitive),
    //Object(String),
    Object(&'a str),
    Array(Box<JavaType<'a>>),
}

impl<'a> core::convert::TryFrom<&'a str> for JavaType<'a> {
    type Error = Error;

    fn try_from(s: &'a str) -> std::result::Result<Self, Self::Error> {
        parser(parse_type)
            .parse(s)
            .map_err(|e| Error::ParseFailed(format!("Failed to parse '{s}': {e}")))
            .map(|(res, tail)| {
                if tail.is_empty() {
                    Ok(res)
                } else {
                    Err(Error::ParseFailed(format!(
                        "Trailing input: '{tail}' while parsing '{s}'"
                    )))
                }
            })?
    }
}

impl fmt::Display for JavaType<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            JavaType::Primitive(ref ty) => ty.fmt(f),
            JavaType::Object(ref name) => write!(f, "L{name};"),
            JavaType::Array(ref ty) => write!(f, "[{ty}"),
        }
    }
}

/// Enum representing any java type that may be used as a return value
///
/// This type intentionally avoids capturing any heap allocated types (to avoid
/// allocations while making JNI method calls) and so it doesn't fully qualify
/// the object or array types with a String like `JavaType::Object` does.
#[allow(missing_docs)]
#[derive(Eq, PartialEq, Debug, Clone)]
pub enum ReturnType {
    Primitive(Primitive),
    Object,
    Array,
}

impl FromStr for ReturnType {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        parser(parse_return)
            .parse(s)
            .map_err(|e| Error::ParseFailed(format!("Failed to parse '{s}': {e}")))
            .map(|(res, tail)| {
                if tail.is_empty() {
                    Ok(res)
                } else {
                    Err(Error::ParseFailed(format!(
                        "Trailing input: '{tail}' while parsing '{s}'"
                    )))
                }
            })?
    }
}

impl fmt::Display for ReturnType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ReturnType::Primitive(ref ty) => ty.fmt(f),
            ReturnType::Object => write!(f, "L;"),
            ReturnType::Array => write!(f, "["),
        }
    }
}

/// A method type signature. This is the structure representation of something
/// like `(Ljava/lang/String;)Z`. Used by the `call_(object|static)_method`
/// functions on jnienv to ensure safety.
#[allow(missing_docs)]
#[derive(Eq, PartialEq, Debug, Clone)]
pub struct TypeSignature<'a> {
    pub args: Vec<JavaType<'a>>,
    pub ret: ReturnType,
}

impl<'a> core::convert::TryFrom<&'a str> for TypeSignature<'a> {
    type Error = Error;

    fn try_from(s: &'a str) -> std::result::Result<Self, Self::Error> {
        parser(parse_sig)
            .parse(s)
            .map_err(|e| Error::ParseFailed(format!("Failed to parse '{s}': {e}")))
            .map(|(res, tail)| {
                if tail.is_empty() {
                    Ok(res)
                } else {
                    Err(Error::ParseFailed(format!(
                        "Trailing input: '{tail}' while parsing '{s}'"
                    )))
                }
            })?
    }
}

impl<'a> fmt::Display for TypeSignature<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(")?;
        for a in &self.args {
            write!(f, "{a}")?;
        }
        write!(f, ")")?;
        write!(f, "{}", self.ret)?;
        Ok(())
    }
}

fn parse_primitive<S: Stream<Token = char>>(input: &mut S) -> StdParseResult<Primitive, S>
where
    S::Error: ParseError<char, S::Range, S::Position>,
{
    let boolean = token('Z').map(|_| Primitive::Boolean);
    let byte = token('B').map(|_| Primitive::Byte);
    let char_type = token('C').map(|_| Primitive::Char);
    let double = token('D').map(|_| Primitive::Double);
    let float = token('F').map(|_| Primitive::Float);
    let int = token('I').map(|_| Primitive::Int);
    let long = token('J').map(|_| Primitive::Long);
    let short = token('S').map(|_| Primitive::Short);
    let void = token('V').map(|_| Primitive::Void);

    (boolean
        .or(byte)
        .or(char_type)
        .or(double)
        .or(float)
        .or(int)
        .or(long)
        .or(short)
        .or(void))
    .parse_stream(input)
    .into()
}

fn parse_array<'a, S>(input: &mut S) -> StdParseResult<JavaType<'a>, S>
where
    S: RangeStream<Token = char, Range = &'a str>,
    S::Error: ParseError<char, S::Range, S::Position>,
{
    let marker = token('[');
    (marker, parser(parse_type))
        .map(|(_, ty)| JavaType::Array(Box::new(ty)))
        .parse_stream(input)
        .into()
}

fn parse_object<'a, S>(input: &mut S) -> StdParseResult<JavaType<'a>, S>
where
    S: RangeStream<Token = char, Range = &'a str>,
    S::Error: ParseError<char, &'a str, S::Position>,
{
    fn is_unqualified(c: char) -> bool {
        // JVMS §4.2.2: '.', ';', '[' and '/' are disallowed in an unqualified name
        !matches!(c, '.' | ';' | '[' | '/')
    }

    // One or more segments separated by '/', never starting or ending with '/'
    let class_body = recognize((
        skip_many1(satisfy(is_unqualified)),
        skip_many(token('/').with(skip_many1(satisfy(is_unqualified)))),
    ));

    (
        token('L'),
        class_body, //.map(|s: &'a str| s.to_owned()),
        token(';'),
    )
        .map(|(_, name, _)| JavaType::Object(name))
        .parse_stream(input)
        .into()
}

fn parse_type<'a, S>(input: &mut S) -> StdParseResult<JavaType<'a>, S>
where
    S: RangeStream<Token = char, Range = &'a str>,
    S::Error: ParseError<char, &'a str, S::Position>,
{
    parser(parse_primitive)
        .map(JavaType::Primitive)
        .or(parser(parse_array))
        .or(parser(parse_object))
        .parse_stream(input)
        .into()
}

fn parse_return<'a, S>(input: &mut S) -> StdParseResult<ReturnType, S>
where
    S: RangeStream<Token = char, Range = &'a str>,
    S::Error: ParseError<char, S::Range, S::Position>,
{
    parser(parse_primitive)
        .map(ReturnType::Primitive)
        .or(parser(parse_array).map(|_| ReturnType::Array))
        .or(parser(parse_object).map(|_| ReturnType::Object))
        .parse_stream(input)
        .into()
}

fn parse_args<'a, S>(input: &mut S) -> StdParseResult<Vec<JavaType<'a>>, S>
where
    S: RangeStream<Token = char, Range = &'a str>,
    S::Error: ParseError<char, S::Range, S::Position>,
{
    between(token('('), token(')'), many(parser(parse_type)))
        .parse_stream(input)
        .into()
}

fn parse_sig<'a, S>(input: &mut S) -> StdParseResult<TypeSignature<'a>, S>
where
    S: RangeStream<Token = char, Range = &'a str>,
    S::Error: ParseError<char, S::Range, S::Position>,
{
    (parser(parse_args), parser(parse_return))
        .map(|(a, r)| TypeSignature { args: a, ret: r })
        .parse_stream(input)
        .into()
}

#[cfg(test)]
mod test {
    use super::*;
    use assert_matches::assert_matches;

    #[test]
    fn test_parser_types() {
        assert_eq!(
            JavaType::try_from("Z").unwrap(),
            JavaType::Primitive(Primitive::Boolean)
        );
        assert_eq!(
            JavaType::try_from("B").unwrap(),
            JavaType::Primitive(Primitive::Byte)
        );
        assert_eq!(
            JavaType::try_from("C").unwrap(),
            JavaType::Primitive(Primitive::Char)
        );
        assert_eq!(
            JavaType::try_from("S").unwrap(),
            JavaType::Primitive(Primitive::Short)
        );
        assert_eq!(
            JavaType::try_from("I").unwrap(),
            JavaType::Primitive(Primitive::Int)
        );
        assert_eq!(
            JavaType::try_from("J").unwrap(),
            JavaType::Primitive(Primitive::Long)
        );
        assert_eq!(
            JavaType::try_from("F").unwrap(),
            JavaType::Primitive(Primitive::Float)
        );
        assert_eq!(
            JavaType::try_from("D").unwrap(),
            JavaType::Primitive(Primitive::Double)
        );
        assert_eq!(
            JavaType::try_from("Ljava/lang/String;").unwrap(),
            JavaType::Object("java/lang/String".into())
        );
        assert_eq!(
            JavaType::try_from("[I").unwrap(),
            JavaType::Array(Box::new(JavaType::Primitive(Primitive::Int)))
        );
        assert_eq!(
            JavaType::try_from("[Ljava/lang/String;").unwrap(),
            JavaType::Array(Box::new(JavaType::Object("java/lang/String".into())))
        );

        assert_matches!(JavaType::try_from(""), Err(_));
        assert_matches!(JavaType::try_from("A"), Err(_));
        // The parser should return an error if the entire input is not consumed (#598)
        assert_matches!(JavaType::try_from("Invalid"), Err(_));
        assert_matches!(JavaType::try_from("II"), Err(_));
        assert_matches!(JavaType::try_from("java/lang/String"), Err(_));
        assert_matches!(JavaType::try_from("Ljava/lang/String"), Err(_));
        assert_matches!(JavaType::try_from("java/lang/String;"), Err(_));
        // Don't allow leading '/' in class names (#212)
        assert_matches!(JavaType::try_from("L/java/lang/String;"), Err(_));
        assert_matches!(JavaType::try_from("L/;"), Err(_));
        assert_matches!(JavaType::try_from("L;"), Err(_));
    }

    #[test]
    fn test_parser_signatures() {
        assert_eq!(
            TypeSignature::try_from("()V").unwrap(),
            TypeSignature {
                args: vec![],
                ret: ReturnType::Primitive(Primitive::Void)
            }
        );
        assert_eq!(
            TypeSignature::try_from("(I)V").unwrap(),
            TypeSignature {
                args: vec![JavaType::Primitive(Primitive::Int)],
                ret: ReturnType::Primitive(Primitive::Void)
            }
        );
        assert_eq!(
            TypeSignature::try_from("(Ljava/lang/String;)I").unwrap(),
            TypeSignature {
                args: vec![JavaType::Object("java/lang/String".into())],
                ret: ReturnType::Primitive(Primitive::Int)
            }
        );
        assert_eq!(
            TypeSignature::try_from("([I)I").unwrap(),
            TypeSignature {
                args: vec![JavaType::Array(Box::new(JavaType::Primitive(
                    Primitive::Int
                )))],
                ret: ReturnType::Primitive(Primitive::Int)
            }
        );
        assert_eq!(
            TypeSignature::try_from("([Ljava/lang/String;)I").unwrap(),
            TypeSignature {
                args: vec![JavaType::Array(Box::new(JavaType::Object(
                    "java/lang/String".into()
                )))],
                ret: ReturnType::Primitive(Primitive::Int)
            }
        );
        assert_eq!(
            TypeSignature::try_from("(I[Ljava/lang/String;Z)I").unwrap(),
            TypeSignature {
                args: vec![
                    JavaType::Primitive(Primitive::Int),
                    JavaType::Array(Box::new(JavaType::Object("java/lang/String".into()))),
                    JavaType::Primitive(Primitive::Boolean),
                ],
                ret: ReturnType::Primitive(Primitive::Int)
            }
        );

        assert_matches!(TypeSignature::try_from(""), Err(_));
        assert_matches!(TypeSignature::try_from("()"), Err(_));
        assert_matches!(TypeSignature::try_from("V"), Err(_));
        assert_matches!(TypeSignature::try_from("(I"), Err(_));
        assert_matches!(TypeSignature::try_from("I)I"), Err(_));
        assert_matches!(TypeSignature::try_from("(I)"), Err(_));
        assert_matches!(TypeSignature::try_from("(Invalid)I"), Err(_));
        // We shouldn't recursively allow method signatures as method argument types (#597)
        assert_matches!(TypeSignature::try_from("((()I)I)I"), Err(_));
        assert_matches!(TypeSignature::try_from("(I)V "), Err(_));
        assert_matches!(TypeSignature::try_from("()java/lang/List"), Err(_));
        assert_matches!(TypeSignature::try_from("(L/java/lang/String)V"), Err(_));
    }
}
