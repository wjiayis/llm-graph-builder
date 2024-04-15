import { ProgressBar } from '@neo4j-ndl/react';

export default function CustomProgressBar({ value }: { value: number }) {
  return <ProgressBar heading='Uploading' className='n-w-40' size='large' value={value}></ProgressBar>;
}
